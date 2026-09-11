"""
Parameterized napari integration tests for the Interface registration workflow.

Tests load project configuration files and simulate the full registration workflow:
1. Project initialization
2. Input/output setup
3. Pair registration
4. Global registration
5. Image fusion

Each test is parameterized to run with different project configurations
(muvis_align_project.yml, muvis_align_project2.yml, etc.).
"""

import logging
import os
import tempfile
import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
import pytest
import yaml
import numpy as np
from qtpy.QtWidgets import QMessageBox

from muvis_align._widget import MainWidget
from muvis_align.ui.Interface import Interface, ViewMode
from muvis_align.MVSRegistration import RegState


@pytest.fixture(autouse=True)
def suppress_completion_dialogs():
    """Keep workflow completion messages from blocking test execution."""
    with patch(
        'muvis_align.ui.Interface.QMessageBox.information'
    ) as mock_information:
        yield mock_information


def get_project_configs():
    """Discover all muvis_align_project*.yml files in tests directory."""
    test_dir = Path(__file__).parent
    configs = sorted(test_dir.glob('muvis_align_project*.yml'))
    if not configs:
        raise FileNotFoundError(
            f"No project config files found in {test_dir}. "
            "Expected muvis_align_project*.yml files."
        )
    return configs


@pytest.fixture(params=get_project_configs(), ids=lambda p: p.name)
def project_config(request):
    """Fixture that provides path to each discovered project configuration file."""
    config_path = request.param
    assert config_path.exists(), f"Project config not found: {config_path}"
    return config_path


@pytest.fixture
def config_data(project_config):
    """Load and parse project configuration YAML."""
    with open(project_config, 'r') as f:
        data = yaml.safe_load(f)
    return data


class TestNapariInterfaceRegistration:
    """Test suite for napari Interface registration workflow with different configs."""

    def test_napari_interface_instantiation(self, make_napari_viewer, project_config):
        """Test that MainWidget and Interface can be instantiated with project config."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget') as mock_viewer_widget:
            with patch.object(viewer.window, 'add_dock_widget'):
                # Mock the overview with necessary attributes
                mock_overview = MagicMock()
                mock_viewer_widget.return_value = mock_overview
                
                # Mock viewer layers
                viewer.layers.clear = MagicMock()
                
                # Mock widget creation functions to avoid magicgui complexity
                with patch('muvis_align.ui.create_widgets.create_project_widget') as mock_proj:
                    with patch('muvis_align.ui.create_widgets.create_template_widgets') as mock_tmpl:
                        mock_proj.return_value = MagicMock()
                        mock_tmpl.return_value = {}
                        
                        try:
                            main_widget = MainWidget(viewer)
                            assert main_widget is not None, "MainWidget should be created"
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            assert False, f"Failed: {type(e).__name__}: {str(e)}\n{tb[:200]}"

    def test_project_config_loading(self, make_napari_viewer, project_config, config_data):
        """Test that project configuration can be loaded into Interface."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget'):
            with patch.object(viewer.window, 'add_dock_widget'):
                main_widget = MainWidget(viewer)
                interface = main_widget.interface
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_copy = Path(tmpdir) / project_config.name
            config_copy.write_text(project_config.read_text())
            
            interface.project_path(str(config_copy))

            assert interface.params_path == str(config_copy)
            assert interface.params is not None
            assert 'registration' in interface.params
            assert 'fusion' in interface.params
            assert 'input_output' in interface.params
            assert 'pre_processing' in interface.params

            # project_path() -> update_input_output_path() -> init_logging() opens a
            # FileHandler on a log file inside tmpdir - it must be closed before this
            # block exits, or Windows refuses to delete the still-open file on cleanup
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                logging.getLogger().removeHandler(handler)

    def test_project_params_structure(self, config_data):
        """Test that project configuration has expected structure."""
        assert isinstance(config_data, dict), "Config should be a dictionary"
        
        required_sections = ['registration', 'fusion', 'input_output']
        for section in required_sections:
            assert section in config_data, f"Missing required section: {section}"
        
        registration = config_data['registration']
        assert 'method' in registration
        assert 'pairing' in registration
        assert 'transform_type' in registration
        assert 'operation' in registration
        
        fusion = config_data['fusion']
        assert 'method' in fusion
        assert 'spacing' in fusion
        assert 'tile_size' in fusion
        
        input_output = config_data['input_output']
        assert 'input_path' in input_output
        assert 'output_path' in input_output

    def test_interface_reset(self, make_napari_viewer, project_config):
        """Test that Interface reset clears state properly."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget'):
            with patch.object(viewer.window, 'add_dock_widget'):
                main_widget = MainWidget(viewer)
                interface = main_widget.interface
        
        interface.reset()
        
        assert hasattr(interface, 'source_metadata'), "Should have source_metadata attribute"
        assert interface.source_metadata == {}, "source_metadata should be empty after reset"
        assert interface.view_mode is None, "view_mode should be None after reset"
        assert interface.selected_shape_index is None, "selected_shape_index should be None after reset"
        assert hasattr(interface.reg, 'state'), "reg should have state attribute"

    def test_interface_tab_management(self, make_napari_viewer, project_config):
        """Test tab enabling/disabling based on registration state."""
        viewer = make_napari_viewer()
        enable_tabs_mock = MagicMock()
        select_tab_mock = MagicMock()
        
        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(
                viewer,
                MagicMock(),
                enable_tabs_mock,
                select_tab_mock,
                verbose=False
            )
        
        # Initial state: all tabs disabled except first
        interface.enable_tabs(False, 1)
        enable_tabs_mock.assert_called_with(False, 1)
        
        # After pair registration: enable up to tab 3
        interface.enable_tabs(True, 3)
        enable_tabs_mock.assert_called_with(True, 3)
        
        # Select specific tab
        interface.select_tab(2)
        select_tab_mock.assert_called_with(2)

    def test_interface_viewer_management(self, make_napari_viewer, project_config):
        """Test napari viewer layer management (clear, add)."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        # Add some test layers
        viewer.add_image(np.random.random((100, 100)), name='test1')
        viewer.add_image(np.random.random((100, 100)), name='test2')
        assert len(viewer.layers) == 2
        
        # Test clear
        interface._clear_napari_view(viewer)
        assert len(viewer.layers) == 0

    def test_interface_view_modes(self, make_napari_viewer, project_config):
        """Test view mode transitions."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        assert interface.view_mode is None
        
        interface.view_mode = ViewMode.OVERVIEW
        assert interface.view_mode == ViewMode.OVERVIEW
        
        interface.view_mode = ViewMode.PAIRS
        assert interface.view_mode == ViewMode.PAIRS
        
        interface.view_mode = ViewMode.FEATURES
        assert interface.view_mode == ViewMode.FEATURES
        
        interface.view_mode = ViewMode.FUSED
        assert interface.view_mode == ViewMode.FUSED

    def test_registration_params_validation(self, config_data):
        """Validate registration parameters in config."""
        registration = config_data['registration']
        
        assert registration['method'] in ['sift', 'orb', 'akaze']
        assert registration['pairing'] in ['orthogonal', 'all']
        assert registration['transform_type'] in ['rigid', 'affine']
        assert registration['operation'] == 'register'
        assert isinstance(registration['max_keypoints'], int)
        assert isinstance(registration['ransac_iterations'], int)

    def test_fusion_params_validation(self, config_data):
        """Validate fusion parameters in config."""
        fusion = config_data['fusion']
        
        assert fusion['method'] in ['average', 'max', 'min']
        assert fusion['spacing'] in ['mean', 'min']
        assert isinstance(fusion['tile_size'], str)
        assert fusion['ome_version'] in ['0.4', '0.5']

    def test_input_output_params_validation(self, config_data):
        """Validate input/output parameters in config."""
        input_output = config_data['input_output']
        
        assert isinstance(input_output['input_path'], str)
        assert isinstance(input_output['output_path'], str)
        assert isinstance(input_output['overwrite'], bool)

    def test_preprocessing_params_validation(self, config_data):
        """Validate pre-processing parameters in config."""
        preprocessing = config_data.get('pre_processing', {})
        
        assert 'scale' in preprocessing
        assert isinstance(preprocessing['scale'], (int, float))
        assert preprocessing['scale'] > 0

    @patch('muvis_align.ui.Interface.QMessageBox.question')
    def test_interface_pair_registration_mock(
        self, mock_question, make_napari_viewer, project_config
    ):
        """Test pair_registration method with mocked UI components and bbox handling."""
        viewer = make_napari_viewer()
        mock_question.return_value = True  # Simulate "Yes" click
        
        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        with patch.object(interface.reg, 'is_global_registered', return_value=False):
            with patch.object(interface.reg, 'is_pairs_registered', return_value=False):
                with patch('muvis_align.ui.Interface.NapariMVSProgress'):
                    with patch('muvis_align.ui.Interface.NapariDaskProgress'):
                        with patch('muvis_align.ui.Interface.TemporarilyDisabledWidgets'):
                            with patch('muvis_align.ui.Interface.VisibleActivityDock'):
                                with patch.object(interface, 'get_all_widgets', return_value={}):
                                    # Create mock bbox DataArray WITHOUT 't' dimension (this is the key test)
                                    import xarray as xr
                                    import networkx as nx
                                    mock_bbox = xr.DataArray(
                                        [[1, 2], [3, 4]],
                                        dims=['x_in', 'x_out'],
                                        coords={'x_in': [0, 1], 'x_out': [0, 1]}
                                    )
                                    
                                    with patch.object(interface.reg, 'register_pairs', return_value={
                                        'pair_mappings': {},
                                        'metrics': {'pairs': {}}
                                    }):
                                        # Mock nx.get_edge_attributes to return bbox without 't' dimension
                                        with patch('networkx.get_edge_attributes') as mock_get_attrs:
                                            mock_get_attrs.return_value = {('key1', 'key2'): mock_bbox}
                                            
                                            with patch.object(interface.reg, 'save_pair_mappings'):
                                                with patch.object(interface, 'update_registered'):
                                                    # This should not raise KeyError
                                                    interface.pair_registration()

    @patch('muvis_align.ui.Interface.QMessageBox.question')
    def test_interface_registration_process_mock(
        self, mock_question, make_napari_viewer, project_config
    ):
        """Test registration_process method with mocked UI components."""
        viewer = make_napari_viewer()
        mock_question.return_value = True  # Simulate "Yes" click
        
        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
            interface.params = {'registration': {'operation': 'register'}}
        
        with patch.object(interface.reg, 'is_pairs_registered', return_value=True):
            with patch.object(interface.reg, 'is_global_registered', return_value=False):
                with patch.object(interface.reg, 'register_global', return_value={
                    'mappings': {},
                    'metrics': {}
                }):
                    with patch('muvis_align.ui.Interface.NapariDaskProgress'):
                        with patch('muvis_align.ui.Interface.TemporarilyDisabledWidgets'):
                            with patch('muvis_align.ui.Interface.VisibleActivityDock'):
                                with patch.object(interface, 'get_all_widgets', return_value={}):
                                    with patch.object(interface.reg, 'save_mappings'):
                                        with patch.object(interface.reg, 'save_metrics'):
                                            with patch.object(interface, 'enable_tabs'):
                                                with patch.object(interface, 'update_registered'):
                                                    interface.registration_process()

    @patch('muvis_align.ui.Interface.QMessageBox.question')
    def test_interface_fusion_process_mock(
        self, mock_question, make_napari_viewer, project_config
    ):
        """Test fusion_process method with mocked UI components."""
        viewer = make_napari_viewer()
        mock_question.return_value = True  # Simulate "Yes" click
        
        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
            interface.params = {
                'registration': {'operation': 'register'},
                'fusion': {
                    'method': 'average',
                    'spacing': 'mean',
                    'tile_size': '1024,1024',
                    'ome_version': '0.5'
                }
            }
        
        with patch.object(interface.reg, 'is_fused', return_value=False):
            with patch('muvis_align.ui.Interface.NapariMVSProgress'):
                with patch('muvis_align.ui.Interface.TemporarilyDisabledWidgets'):
                    with patch('muvis_align.ui.Interface.VisibleActivityDock'):
                        with patch.object(interface, 'get_all_widgets', return_value={}):
                            with patch.object(interface.reg, 'fuse', return_value=(MagicMock(), None)):
                                with patch.object(interface, '_clear_napari_view'):
                                    with patch.object(interface, '_napari_view_add_image'):
                                        interface.fusion_process()

    def test_registration_state_transitions(self, make_napari_viewer, project_config):
        """Test valid registration state transitions."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        # Verify state progression methods
        assert not interface.reg.is_pairs_registered()
        assert not interface.reg.is_global_registered()
        assert not interface.reg.is_fused()
        
        # Simulate state transitions
        interface.reg.state = RegState.PAIRS_REG
        assert interface.reg.is_pairs_registered()
        assert not interface.reg.is_global_registered()
        
        interface.reg.state = RegState.GLOBAL_REG
        assert interface.reg.is_pairs_registered()
        assert interface.reg.is_global_registered()
        
        interface.reg.state = RegState.FUSED
        assert interface.reg.is_pairs_registered()
        assert interface.reg.is_global_registered()
        assert interface.reg.is_fused()

    def test_metrics_methods_available(self, make_napari_viewer, project_config):
        """Test that expected metrics methods are available."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        assert interface.metrics_methods == ['ncc', 'ssim', 'onmi']
        assert len(interface.metrics_methods) > 0
        for metric in interface.metrics_methods:
            assert isinstance(metric, str)

    def test_interface_initialization_with_template(self, make_napari_viewer, project_config):
        """Test that Interface properly initializes with project template."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())
        
        assert interface.raw_template is not None
        assert interface.template is not None
        assert isinstance(interface.template, dict)

    @patch('muvis_align.ui.Interface.QMessageBox.question')
    def test_modify_pair_registration_with_bbox(self, mock_question, make_napari_viewer, project_config):
        """Test modify_pair_registration with bbox handling (no 't' dimension)."""
        viewer = make_napari_viewer()
        mock_question.return_value = QMessageBox.Yes  # Simulate "Yes" click

        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())

        interface.view_mode = ViewMode.PAIRS
        interface.pair_indices = ('key1', 'key2')
        interface.reg.pairs_graph = object()
        interface.reg.source_transform_key = 'source_metadata'
        
        # Mock the temp_widget_state that gets called in modify_pair_registration
        interface.temp_widget_state = MagicMock()

        with patch.object(interface, 'calc_mod_pair_transform') as mock_calc:
            with patch('networkx.get_edge_attributes') as mock_get_attrs:
                # Create mock bbox DataArray WITHOUT 't' dimension
                import xarray as xr
                mock_bbox = xr.DataArray(
                    [[1, 2], [3, 4]],
                    dims=['x_in', 'x_out'],
                    coords={'x_in': [0, 1], 'x_out': [0, 1]}
                )
                
                # Mock transform with 't' dimension for the pair_transforms
                mock_transform_with_t = xr.DataArray(
                    [[[1, 0], [0, 1], [0, 0]]],
                    dims=['t', 'rows', 'cols'],
                    coords={'t': [0]}
                )
                
                # Set up return values - first call gets pair_transforms, second gets qualities, third gets bboxes
                mock_get_attrs.side_effect = [
                    {interface.pair_indices: mock_transform_with_t},  # pair_transforms
                    {interface.pair_indices: 0.95},  # qualities
                    {interface.pair_indices: mock_bbox}  # bboxes (without 't' dimension)
                ]
                
                mock_calc.return_value = mock_transform_with_t.sel(t=0)
                
                with patch('networkx.set_edge_attributes'):
                    with patch.object(interface.reg, 'save_pair_mappings') as mock_save:
                        with patch.object(interface, 'update_registered'):
                            # This should not raise KeyError
                            interface.modify_pair_registration()
                            
                            # Verify save_pair_mappings was called
                            assert mock_save.called

    def test_global_registration_with_dimension_mismatch(self, make_napari_viewer, project_config):
        """Test update_registered handles msims with transforms that include a t dimension."""
        viewer = make_napari_viewer()

        with patch('muvis_align._widget.ViewerWidget'):
            interface = Interface(viewer, MagicMock(), MagicMock(), MagicMock())

        # Mock the missing reg_transform_key attribute on MVSRegistration
        interface.reg.reg_transform_key = 'registered'
        
        with patch('muvis_align.ui.Interface.si_utils.get_tranform_keys_from_sim', return_value=['registered']):
            with patch.object(interface, 'populate_coordinate_systems') as mock_pop_coord:
                with patch.object(interface, 'populate_metadata_table') as mock_pop_meta:
                    with patch.object(interface, 'populate_metrics_table') as mock_pop_metrics:
                        with patch.object(interface, 'update_views') as mock_views:
                            # This should not raise KeyError about dimension mismatch
                            interface.update_registered()

                            assert mock_pop_coord.called
                            assert mock_pop_meta.called
                            assert mock_pop_metrics.called
                            mock_views.assert_called_once_with(transform_key=None,
                                                              progress_factory=None)


class TestProjectConfigurationFiles:
    """Test suite for project configuration file validation."""

    def test_all_configs_valid_yaml(self, project_config):
        """Test that all project configs are valid YAML."""
        with open(project_config, 'r') as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_config_file_exists(self, project_config):
        """Test that config file exists and is readable."""
        assert project_config.exists()
        assert project_config.is_file()
        assert os.access(project_config, os.R_OK)

    def test_config_has_required_keys(self, config_data):
        """Test that config has all required top-level keys."""
        required_keys = ['registration', 'fusion', 'input_output']
        for key in required_keys:
            assert key in config_data, f"Missing required key: {key}"


class TestMainWidgetIntegration:
    """Test suite for MainWidget integration with different configs."""

    def test_main_widget_creation(self, make_napari_viewer, project_config):
        """Test MainWidget creation with napari viewer."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget'):
            with patch.object(viewer.window, 'add_dock_widget'):
                widget = MainWidget(viewer)
        
        assert widget is not None
        assert hasattr(widget, 'interface')
        assert hasattr(widget, 'viewer')
        assert widget.viewer is viewer

    def test_main_widget_tab_creation(self, make_napari_viewer, project_config):
        """Test that MainWidget creates tabs correctly."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget'):
            with patch.object(viewer.window, 'add_dock_widget'):
                widget = MainWidget(viewer)
        
        assert widget.count() > 0
        assert len(widget.tab_labels) > 0
        assert 'project' in widget.tab_labels

    def test_main_widget_tab_disabled_initially(self, make_napari_viewer, project_config):
        """Test that non-project tabs are disabled initially."""
        viewer = make_napari_viewer()
        
        with patch('muvis_align._widget.ViewerWidget'):
            with patch.object(viewer.window, 'add_dock_widget'):
                widget = MainWidget(viewer)
        
        # Project tab should be enabled
        assert widget.isTabEnabled(0)
        
        # Other tabs should be disabled initially
        if widget.count() > 1:
            assert not widget.isTabEnabled(1)


# Use the canonical package name for unit-level coverage.  The older integration
# tests above intentionally retain their historical ``muvis_align`` imports.
interface_module = importlib.import_module("muvis_align.ui.Interface")
CanonicalInterface = interface_module.Interface
CanonicalViewMode = interface_module.ViewMode
CanonicalRegState = interface_module.RegState


@pytest.fixture
def bare_interface():
    interface = CanonicalInterface.__new__(CanonicalInterface)
    interface.reg = MagicMock()
    interface.verbose = False
    interface.enable_plugin_widget = None
    interface.extra_metadata = {}
    interface.param_widgets = {}
    interface.reg.file_labels = ["image-0"]
    interface.view_msims = [
        SimpleNamespace(dims=("z", "y", "x"), sizes={"z": 2})
    ]
    return interface


def test_change_param_updates_nested_value_and_writes(bare_interface):
    bare_interface.params = {}
    bare_interface.write_params = MagicMock()

    bare_interface.change_param("registration.method", "phase")

    assert bare_interface.params == {
        "registration": {"method": "phase"}
    }
    bare_interface.write_params.assert_called_once_with()


def test_change_param_relativizes_input_output_paths(bare_interface, tmp_path):
    """A file dialog (or the FileEdit widget itself) always reports an absolute path -
    change_param() must convert it back to relative-to-project-dir before storing it, so the
    project file keeps portable relative paths instead of being silently rewritten absolute."""
    bare_interface.params = {}
    bare_interface.write_params = MagicMock()
    bare_interface.params_path = str(tmp_path / "project.yml")
    absolute_input = str(tmp_path / "data" / "input")

    bare_interface.change_param("input_output.input_path", absolute_input)

    assert bare_interface.params["input_output"]["input_path"] == "data/input"


def test_change_param_leaves_other_params_untouched_by_relativizing(bare_interface, tmp_path):
    bare_interface.params = {}
    bare_interface.write_params = MagicMock()
    bare_interface.params_path = str(tmp_path / "project.yml")

    bare_interface.change_param("registration.method", str(tmp_path / "not-a-path-param"))

    assert bare_interface.params["registration"]["method"] == str(tmp_path / "not-a-path-param").replace('\\', '/')


def test_get_project_dir_returns_none_before_project_loaded(bare_interface):
    assert not hasattr(bare_interface, "params_path")
    assert bare_interface.get_project_dir() is None


def test_update_input_output_path_displays_stored_value_as_is(bare_interface, tmp_path):
    """input/output paths are stored relative to the project directory - the widgets must
    display that exact relative text, not an absolute path, so a loaded project still looks
    relative in the UI. FileEdit.set_value() would force it absolute, so the display path must
    be written straight into the widget's inner line edit instead."""
    bare_interface.params_path = str(tmp_path / "project.yml")
    bare_interface.params = {
        "input_output": {"input_path": "data/input", "output_path": "results"}
    }
    input_widget = MagicMock()
    output_widget = MagicMock()
    bare_interface.param_widgets = {
        "input_output.input_path": input_widget,
        "input_output.output_path": output_widget,
    }

    bare_interface.update_input_output_path()

    assert input_widget.widget.line_edit.value == "data/input"
    assert output_widget.widget.line_edit.value == "results"
    input_widget.set_value.assert_not_called()
    output_widget.set_value.assert_not_called()


def test_update_input_output_path_falls_back_to_set_value_without_line_edit(bare_interface, tmp_path):
    """A widget without an inner line_edit (e.g. not a FileEdit) falls back to the normal
    set_value() path instead of erroring."""
    bare_interface.params_path = str(tmp_path / "project.yml")
    bare_interface.params = {
        "input_output": {"input_path": "data/input", "output_path": ""}
    }
    input_widget = MagicMock()
    input_widget.widget = SimpleNamespace()  # no line_edit attribute
    bare_interface.param_widgets = {
        "input_output.input_path": input_widget,
        "input_output.output_path": MagicMock(),
    }

    bare_interface.update_input_output_path()

    input_widget.set_value.assert_called_once_with("data/input")


def test_input_output_process_resolves_relative_paths_before_reg_init(
    bare_interface, tmp_path, mocked_activity_contexts
):
    """input_path/output_path are stored relative to the project directory - MVSRegistration
    resolves a relative path against the process's cwd (not the project dir), so
    input_output_process() must resolve them to absolute paths before calling reg.init()."""
    bare_interface.viewer = MagicMock()
    bare_interface.params_path = str(tmp_path / "project.yml")
    bare_interface.params = {
        "input_output": {
            "input_path": "data/input",
            "output_path": "results",
            "overwrite": True,
        }
    }
    bare_interface.reg.is_initialised.return_value = False
    bare_interface.need_source_reinit = False
    bare_interface.reg.init.return_value = True
    bare_interface.update_metadata_source = MagicMock(return_value=True)
    bare_interface.populate_image_selection = MagicMock()
    bare_interface._load_saved_progress = MagicMock()
    bare_interface._show_loaded_project = MagicMock()

    bare_interface.input_output_process()

    expected_input = str(tmp_path / "data" / "input").replace('\\', '/')
    expected_output = str(tmp_path / "results").replace('\\', '/') + '/'
    bare_interface.reg.init.assert_called_once_with(
        input_path=expected_input,
        output_path=expected_output,
        overwrite=True,
        verbose=False,
    )


def test_get_all_widgets_excludes_widgets_on_disabled_tabs(bare_interface):
    """A widget on a currently disabled tab always reads .enabled == False, so if
    modify_pair_registration snapshotted and restored it via get_all_widgets, it would stay
    disabled even after its tab becomes enabled later. get_all_widgets must exclude it instead."""
    bare_interface.param_widgets = {
        "registration.method": SimpleNamespace(widget="reg-widget"),
        "fusion.method": SimpleNamespace(widget="fusion-widget"),
    }
    bare_interface.is_tab_enabled = lambda section_id: section_id != "fusion"

    all_widgets = bare_interface.get_all_widgets()

    assert all_widgets == {"registration.method": "reg-widget"}


def test_modify_pair_registration_disables_other_tabs_and_restores_them(
    bare_interface, monkeypatch
):
    """Entering pair-modification mode must disable every other tab (not registration - its own
    widgets are already disabled via get_all_widgets) so the user can't navigate away
    mid-adjustment, and restore each tab's prior enabled state on exit."""
    import xarray as xr

    bare_interface.view_mode = None
    bare_interface.viewer = MagicMock()
    bare_interface.template = {
        "input_output": [], "pre_processing": [], "registration": [], "fusion": []
    }
    tab_states = {
        "project": True, "input_output": True, "pre_processing": True,
        "registration": True, "fusion": False,
    }
    bare_interface.is_tab_enabled = lambda section_id: tab_states[section_id]
    bare_interface.enable_tab = MagicMock(
        side_effect=lambda section_id, enabled: tab_states.__setitem__(section_id, enabled)
    )
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface.param_widgets = {
        "registration.reg_preview_image1": SimpleNamespace(get_value=lambda: "image-0"),
        "registration.reg_preview_image2": SimpleNamespace(get_value=lambda: "image-0"),
    }
    bare_interface.reg.file_labels = ["image-0"]
    bare_interface.reg.register_msims = ["msim"]
    transform = xr.DataArray(
        np.eye(3).reshape(1, 3, 3), dims=["t", "x_in", "x_out"], coords={"t": [0]}
    )
    monkeypatch.setattr(
        interface_module.nx, "get_edge_attributes", lambda *_: {(0, 0): transform}
    )
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._napari_view_add_image = MagicMock()
    bare_interface.update_pair_metrics = MagicMock()

    bare_interface.modify_pair_registration()

    disabled_ids = [call.args[0] for call in bare_interface.enable_tab.call_args_list]
    assert set(disabled_ids) == {"project", "input_output", "pre_processing", "fusion"}
    assert tab_states == {
        "project": False, "input_output": False, "pre_processing": False,
        "registration": True, "fusion": False,
    }

    bare_interface.enable_tab.reset_mock()
    bare_interface.update_registered = MagicMock()
    monkeypatch.setattr(
        interface_module.QMessageBox, "question", lambda *_: interface_module.QMessageBox.No
    )

    bare_interface.modify_pair_registration()

    assert tab_states == {
        "project": True, "input_output": True, "pre_processing": True,
        "registration": True, "fusion": False,
    }


def test_tab_changed_clears_feature_view_and_stops_timer(bare_interface):
    bare_interface.viewer = MagicMock()
    bare_interface.view_mode = CanonicalViewMode.FEATURES
    bare_interface.pair_metrics_timer = MagicMock()
    bare_interface._clear_napari_view = MagicMock()

    bare_interface.tab_changed("fusion")

    bare_interface._clear_napari_view.assert_called_once_with(
        bare_interface.viewer
    )
    bare_interface.pair_metrics_timer.stop.assert_called_once_with()
    assert bare_interface.view_mode is None


@pytest.mark.parametrize(
    ("method_name", "section", "axis"),
    [
        ("source_position_z", "position", "z"),
        ("source_position_y", "position", "y"),
        ("source_position_x", "position", "x"),
        ("source_scale_z", "scale", "z"),
        ("source_scale_y", "scale", "y"),
        ("source_scale_x", "scale", "x"),
    ],
)
def test_source_metadata_setters(
    bare_interface, method_name, section, axis
):
    bare_interface.source_metadata = {}

    getattr(bare_interface, method_name)(2.5)

    assert bare_interface.source_metadata[section][axis] == 2.5


def test_source_rotation_sets_valid_value(bare_interface):
    bare_interface.source_metadata = {}

    bare_interface.source_rotation(12.5)

    assert bare_interface.source_metadata["rotation"] == 12.5


@pytest.mark.parametrize("exists", [True, False], ids=["existing", "new"])
def test_project_path_handles_existing_and_new_projects(
    bare_interface, monkeypatch, exists
):
    bare_interface.template = {"template": True}
    bare_interface.reset = MagicMock()
    bare_interface.update_widgets = MagicMock()
    bare_interface.write_params = MagicMock()
    bare_interface.update_input_output_path = MagicMock()
    monkeypatch.setattr(interface_module.os.path, "exists", lambda _: exists)
    monkeypatch.setattr(
        interface_module,
        "get_template_params",
        lambda _: {"input_output": {}},
    )
    monkeypatch.setattr(
        interface_module, "read_params", lambda _: {"registration": {}}
    )
    monkeypatch.setattr(
        interface_module,
        "update_params",
        lambda defaults, loaded: defaults | loaded,
    )

    bare_interface.project_path("project.yml")

    bare_interface.reset.assert_called_once_with()
    assert bare_interface.params_path == "project.yml"
    # update_input_output_path() must run for both an existing and a brand-new project - it
    # resolves the stored (relative-to-project-dir) input/output paths for display, which
    # update_widgets() deliberately skips
    bare_interface.update_input_output_path.assert_called_once_with()
    if exists:
        bare_interface.update_widgets.assert_called_once_with()
        bare_interface.write_params.assert_not_called()
    else:
        bare_interface.write_params.assert_called_once_with()
        bare_interface.update_widgets.assert_not_called()


def test_populate_choices_and_image_selection(bare_interface):
    channel_widget = MagicMock()
    coordinate_widget = MagicMock()
    image1_widget = MagicMock()
    image2_widget = MagicMock()
    bare_interface.param_widgets = {
        "registration.channel": channel_widget,
        "input_output.coordinate_system": coordinate_widget,
        "registration.reg_preview_image1": image1_widget,
        "registration.reg_preview_image2": image2_widget,
    }
    bare_interface.reg.sources = [
        SimpleNamespace(
            get_channels=lambda: [{"label": "red"}, {"label": "green"}]
        )
    ]
    bare_interface.reg.file_labels = ["left", "right"]

    bare_interface.populate_channels()
    bare_interface.populate_coordinate_systems(
        ["source_metadata", "registered"]
    )
    bare_interface.populate_image_selection()

    assert channel_widget.set_choices.call_args.args[0] == {
        "red": "red",
        "green": "green",
    }
    assert coordinate_widget.set_choices.call_args.args[0] == {
        "source_metadata": "Source metadata",
        "registered": "Registered",
    }
    image1_widget.set_value.assert_called_once_with(
        "left", choices=["left", "right"]
    )
    image2_widget.set_value.assert_called_once_with(
        "right", choices=["left", "right"]
    )


@pytest.mark.parametrize(
        ("transforms", "expected"),
        [
            (["source_metadata"], "source_metadata"),
            (["source_metadata", "transform"], "transform"),
            (["registered", "transform"], "registered"),
            ([], None),
        ],
)
def test_get_best_transform_key(
    bare_interface, monkeypatch, transforms, expected
):
    bare_interface.reg.reg_transform_key = "registered"
    bare_interface.reg.source_transform_key = "source_metadata"
    monkeypatch.setattr(
        interface_module, "get_transforms", lambda _: transforms
    )

    assert bare_interface.get_best_transform_key() == expected


def test_update_views_adds_enabled_preview_layers(
    bare_interface, monkeypatch
):
    bare_interface.viewer = MagicMock()
    bare_interface.overview = MagicMock()
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"z": 2})]
    bare_interface.reg.positions = [{"z": 0}]
    bare_interface.reg.fileset_label = "sample"
    bare_interface.get_best_transform_key = MagicMock(
        return_value="registered"
    )
    bare_interface._clear_napari_view = MagicMock()
    shapes = [np.zeros((4, 2))]
    shape_data = (shapes, ["0"], ["image-0"], [(1, 1, 1)])
    image_data = object()
    bare_interface._create_napari_shapes = MagicMock(
        return_value=shape_data
    )
    bare_interface._create_napari_data = MagicMock(
        return_value=image_data
    )
    bare_interface._napari_view_add_fused_data = MagicMock()
    bare_interface._update_view_add_shapes = MagicMock()
    monkeypatch.setattr(
        interface_module.si_utils,
        "get_origin_from_sim",
        lambda _: {"z": 0},
    )
    monkeypatch.setattr(
        interface_module, "get_msim_image0", lambda msim: msim
    )

    bare_interface.update_views(show_preprocessed=True)

    assert bare_interface._clear_napari_view.call_args_list == [
        call(bare_interface.viewer),
        call(bare_interface.overview),
    ]
    assert bare_interface._create_napari_shapes.call_args_list == [
        call("registered", force_2d=False),
        call("registered", force_2d=True),
    ]
    bare_interface._create_napari_data.assert_called_once_with(
        "registered",
        show_preprocessed=True,
    )
    bare_interface._napari_view_add_fused_data.assert_called_once_with(
        bare_interface.viewer, image_data, "sample data", cheap=True
    )
    expected_shape_call = (
        shapes, ["0"], ["image-0"], [(1, 1, 1)], "sample shapes"
    )
    bare_interface._update_view_add_shapes.assert_any_call(
        bare_interface.viewer,
        *expected_shape_call,
    )
    bare_interface._update_view_add_shapes.assert_any_call(
        bare_interface.overview,
        *expected_shape_call,
    )
    assert bare_interface._update_view_add_shapes.call_count == 2
    assert bare_interface.view_mode is CanonicalViewMode.OVERVIEW


def test_update_views_detects_multi_z_from_view_msims(
    bare_interface, monkeypatch
):
    bare_interface.viewer = MagicMock()
    bare_interface.overview = MagicMock()
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"y": 10, "x": 10})]
    bare_interface.reg.positions = [{"z": 0}, {"z": 1}]
    bare_interface._create_napari_shapes = MagicMock(
        return_value=([], [], [], [])
    )
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._update_view_add_shapes = MagicMock()

    bare_interface.update_views(transform_key="source_metadata", show_images=False)

    bare_interface._create_napari_shapes.assert_called_once_with(
        "source_metadata", force_2d=True
    )


def test_update_napari_shapes_adds_3d_box_with_overlap_metadata(
    bare_interface, monkeypatch
):
    """A 3D box is shown as its own 6 flat quad faces (an opaque, quality-colored box - a
    single 'polygon' can't represent a whole non-planar box) plus one edge-only 'path'
    wireframe, since napari-bbox (which drew real 3D boxes) is incompatible with current
    napari. Distinct per-corner values (rather than all-zeros/all-ones) so a face/wire
    indexing mistake would actually be caught."""
    viewer = MagicMock()
    image_shape = np.arange(24, dtype=float).reshape(8, 3)
    overlap_shape = np.arange(24, 48, dtype=float).reshape(8, 3)
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"z": 2})]
    bare_interface.reg.positions = [{"z": 0}]
    bare_interface.reg.get_metrics.return_value = 0.75
    create_shapes = MagicMock(return_value=[image_shape])
    create_overlaps = MagicMock(
        return_value=([overlap_shape], [np.array([0, 0])])
    )
    monkeypatch.setattr(
        interface_module.si_utils,
        "get_origin_from_sim",
        lambda _: {"z": 0},
    )
    monkeypatch.setattr(
        interface_module, "get_msim_image0", lambda msim: msim
    )
    monkeypatch.setattr(interface_module, "create_image_shapes", create_shapes)
    monkeypatch.setattr(
        interface_module, "create_overlap_shapes", create_overlaps
    )
    monkeypatch.setattr(
        interface_module, "metric_to_rgb", lambda _: (0.1, 0.2, 0.3)
    )

    shape_data = bare_interface._create_napari_shapes("registered")
    bare_interface._update_view_add_shapes(viewer, *shape_data, "boxes")

    assert viewer.add_shapes.call_count == 1
    args, kwargs = viewer.add_shapes.call_args
    shape_data_out = args[0]
    edge_path = [0, 1, 2, 3, 0, 4, 7, 3, 2, 6, 7, 4, 5, 6, 2, 1, 5]
    expected_wire_width = np.ptp(
        np.concatenate([image_shape, overlap_shape]), axis=0
    ).max() * 0.005

    assert kwargs["shape_type"] == ["polygon"] * 12 + ["path"] * 2
    # napari/napari#6860: a 3D 'polygon' face only renders if it's axis-orthogonal, so the
    # fill is built from each box's own axis-aligned bounding box rather than its (possibly
    # non-axis-aligned) true corners - check the 6 faces per box collectively bound the same
    # extent as the box's real corners, rather than an exact per-vertex index match.
    image_faces = np.concatenate([np.asarray(q) for q in shape_data_out[:6]])
    overlap_faces = np.concatenate([np.asarray(q) for q in shape_data_out[6:12]])
    np.testing.assert_allclose(image_faces.min(axis=0), image_shape.min(axis=0))
    np.testing.assert_allclose(image_faces.max(axis=0), image_shape.max(axis=0))
    np.testing.assert_allclose(overlap_faces.min(axis=0), overlap_shape.min(axis=0))
    np.testing.assert_allclose(overlap_faces.max(axis=0), overlap_shape.max(axis=0))
    # the wireframe keeps the true (possibly non-axis-aligned) corners, since edges already
    # render fine in 3D regardless of orientation.
    np.testing.assert_allclose(shape_data_out[12], image_shape[edge_path])
    np.testing.assert_allclose(shape_data_out[13], overlap_shape[edge_path])

    # the quality-based color is carried onto every face of its own box; the wireframe
    # paths get a placeholder color, since a 'path' has no face to color and its edge is
    # drawn regardless of face_color. face_color entries may be numpy arrays (metric_to_rgb's
    # output) - compare by value rather than with == (ambiguous truth value for a list
    # containing arrays).
    face_color = kwargs["face_color"]
    for i in range(6):
        np.testing.assert_allclose(face_color[i], (1, 1, 1))
        np.testing.assert_allclose(face_color[6 + i], (0.1, 0.2, 0.3))
    np.testing.assert_allclose(face_color[12], (0, 0, 0))
    np.testing.assert_allclose(face_color[13], (0, 0, 0))
    # every entry must be the same length - napari can't build one color array from mixed
    # 3- and 4-tuples and silently falls back to plain white for the whole layer instead
    assert len({len(c) for c in face_color}) == 1
    edge_color = kwargs["edge_color"]
    assert len({len(c) for c in edge_color}) == 1
    np.testing.assert_allclose(edge_color[:12], [(0, 0, 0)] * 12)
    np.testing.assert_allclose(edge_color[12:], [(0, 1, 1)] * 2)
    np.testing.assert_allclose(
        kwargs["edge_width"], [0] * 12 + [expected_wire_width] * 2
    )
    assert kwargs["features"]["refs"] == ["0"] * 6 + ["0 0"] * 6 + ["0", "0 0"]
    assert kwargs["features"]["labels"] == [""] * 12 + ["image-0", ""]


def test_update_napari_shapes_3d_faces_are_axis_aligned_and_wind_outward(
    bare_interface, monkeypatch
):
    """napari/napari#6860: napari's 3D Shapes layer only renders a polygon's face fill if
    that face's own plane is axis-orthogonal - a genuinely oriented/rotated box (as global
    registration can produce) gets no face fill at all except for whichever face happens to
    be axis-aligned. The fill must therefore come from the box's axis-aligned bounding box,
    never its own (here, rotated) corners. Separately, box_faces' index order alone doesn't
    guarantee consistent outward winding even for an axis-aligned box - the per-face
    flip-if-inward correction must still catch and fix that."""
    viewer = MagicMock()
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"z": 2})]
    bare_interface.reg.positions = [{"z": 0}]
    unit_cube = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    theta = np.pi / 6
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotated_box = unit_cube @ rotation.T + np.array([10.0, 20.0, 30.0])

    monkeypatch.setattr(
        interface_module.si_utils, "get_origin_from_sim", lambda _: {"z": 0}
    )
    monkeypatch.setattr(
        interface_module, "get_msim_image0", lambda msim: msim
    )
    monkeypatch.setattr(
        interface_module, "create_image_shapes", lambda *_, **__: [rotated_box]
    )
    monkeypatch.setattr(
        interface_module, "create_overlap_shapes", lambda *_, **__: ([], [])
    )

    shape_data = bare_interface._create_napari_shapes("registered")
    bare_interface._update_view_add_shapes(viewer, *shape_data, "boxes")

    assert viewer.add_shapes.call_count == 1
    args, kwargs = viewer.add_shapes.call_args
    shape_data_out = args[0]
    face_quads = [np.asarray(s) for s, t in zip(shape_data_out, kwargs["shape_type"])
                 if t == "polygon"]
    assert len(face_quads) == 6

    box_min, box_max = rotated_box.min(axis=0), rotated_box.max(axis=0)
    box_center = (box_min + box_max) / 2
    for quad in face_quads:
        normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
        nonzero_axes = np.sum(~np.isclose(normal, 0))
        assert nonzero_axes == 1, f"face normal {normal} is not axis-orthogonal"
        outward = quad.mean(axis=0) - box_center
        assert np.dot(normal, outward) >= 0, "face wound inward - would be backface-culled"

    # the fill's overall extent must still bound the true (rotated) box, not some
    # unrelated or degenerate region
    all_face_points = np.concatenate(face_quads)
    np.testing.assert_allclose(all_face_points.min(axis=0), box_min)
    np.testing.assert_allclose(all_face_points.max(axis=0), box_max)


def test_update_napari_shapes_uses_shapes_layer_for_2d(
    bare_interface, monkeypatch
):
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"y": 10, "x": 10})]
    bare_interface.reg.positions = [{"z": 0}]
    viewer = MagicMock()
    shape = np.zeros((4, 2))
    monkeypatch.setattr(
        interface_module.si_utils, "get_origin_from_sim", lambda _: {}
    )
    monkeypatch.setattr(
        interface_module, "get_msim_image0", lambda msim: msim
    )
    monkeypatch.setattr(
        interface_module, "create_image_shapes", lambda *_, **__: [shape]
    )

    shapes = [shape]
    bare_interface._update_view_add_shapes(
        viewer, shapes, ["0"], ["image-0"], [(1, 1, 1)], "boxes"
    )

    args, kwargs = viewer.add_shapes.call_args
    np.testing.assert_allclose(args[0], [shape])
    assert kwargs["shape_type"] == "polygon"
    assert kwargs["edge_width"] == 0.1
    assert kwargs["features"]["refs"] == ["0"]


def test_update_napari_features_dispatches_all_layer_types(
    bare_interface, monkeypatch
):
    viewer = MagicMock()
    layers = [
        ("image", {"name": "image"}, "image"),
        ("points", {"name": "points"}, "points"),
        ("shapes", {"name": "shapes"}, "shapes"),
    ]
    monkeypatch.setattr(
        interface_module,
        "draw_keypoints_matches_napari",
        lambda *_, **__: layers,
    )

    bare_interface._napari_view_show_features(
        viewer, None, None, None, None, None, None
    )

    viewer.layers.clear.assert_called_once_with()
    viewer.add_image.assert_called_once_with("image", name="image")
    viewer.add_points.assert_called_once_with("points", name="points")
    viewer.add_shapes.assert_called_once_with("shapes", name="shapes")


def test_add_napari_image_applies_color_and_affine_callback(
    bare_interface, monkeypatch
):
    viewer = MagicMock()
    layer = viewer.add_image.return_value
    data = object()
    monkeypatch.setattr(
        interface_module.si_utils,
        "get_spacing_from_sim",
        lambda *_args, **_kwargs: [2, 3],
    )
    monkeypatch.setattr(
        interface_module.si_utils,
        "get_origin_from_sim",
        lambda *_args, **_kwargs: [4, 5],
    )

    result = bare_interface._napari_view_add_image(
        viewer,
        data,
        "image",
        transform="affine",
        color="red",
        affine_event=True,
    )

    assert result is layer
    assert layer.colormap == "red"
    layer.events.affine.connect.assert_called_once_with(
        bare_interface.on_image_data_changed
    )


def test_on_image_data_changed_restarts_metrics_timer(bare_interface):
    bare_interface.pair_metrics_timer = MagicMock()

    bare_interface.on_image_data_changed(object())

    bare_interface.pair_metrics_timer.stop.assert_called_once_with()
    bare_interface.pair_metrics_timer.start.assert_called_once_with()


@pytest.fixture
def mocked_activity_contexts(monkeypatch):
    monkeypatch.setattr(
        interface_module, "NapariMVSProgress", lambda **_: nullcontext()
    )
    monkeypatch.setattr(
        interface_module, "NapariDaskProgress", lambda **_: nullcontext()
    )
    monkeypatch.setattr(
        interface_module,
        "TemporarilyDisabledWidgets",
        lambda _: nullcontext(),
    )
    monkeypatch.setattr(
        interface_module, "VisibleActivityDock", lambda _: nullcontext()
    )


def test_run_pair_registration_serializes_quality_and_time_bbox(
    bare_interface, monkeypatch, mocked_activity_contexts
):
    import xarray as xr

    bare_interface.viewer = MagicMock()
    bare_interface.params = {"registration": {"method": "phase"}}
    bare_interface.metrics_methods = ["ncc"]
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface.reg.register_msims = ["register-msim"]
    bare_interface.reg.pairs_graph = object()
    results = {
        "pair_mappings": {(0, 1): "mapping"},
        "metrics": {
            "pairs": {
                (0, 1): {
                    interface_module.default_transform_key: {
                        interface_module.default_quality_key: 0.9
                    }
                }
            }
        },
    }
    bare_interface.reg.register_pairs.return_value = results
    bbox = xr.DataArray(
        [[[1, 2], [3, 4]]],
        dims=("t", "corner", "axis"),
        coords={"t": [0]},
    )
    monkeypatch.setattr(
        interface_module.nx,
        "get_edge_attributes",
        lambda *_: {(0, 1): bbox},
    )

    actual = bare_interface.run_pair_registration()

    assert actual is results
    bare_interface.reg.register_pairs.assert_called_once_with(
        ["register-msim"],
        params={"method": "phase", "metrics": ["ncc"]},
    )
    bare_interface.reg.save_pair_mappings.assert_called_once_with(
        {(0, 1): "mapping"},
        {(0, 1): 0.9},
        {(0, 1): [[1, 2], [3, 4]]},
    )


def test_run_global_registration_persists_all_results(
    bare_interface, mocked_activity_contexts
):
    bare_interface.viewer = MagicMock()
    bare_interface.params = {"registration": {"method": "phase"}}
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface.reg.pair_msims = ["msim"]
    bare_interface.reg.register_indices = [0]
    results = {
        "mappings": {"image-0": "mapping"},
        "metrics": {"summary": {}},
    }
    bare_interface.reg.register_global.return_value = results

    actual = bare_interface.run_global_registration()

    assert actual is results
    bare_interface.reg.register_global.assert_called_once_with(
        ["msim"],
        register_indices=[0],
        params={"method": "phase"},
    )
    bare_interface.reg.save_mappings.assert_called_once_with(
        results["mappings"]
    )
    bare_interface.reg.save_mappings_csv.assert_called_once_with(
        results["mappings"]
    )
    bare_interface.reg.save_metrics.assert_called_once_with(
        results["metrics"]
    )


def _stub_preview_registration_deps(bare_interface, monkeypatch, label1="image-0", label2="image-1"):
    bare_interface.viewer = MagicMock()
    bare_interface.param_widgets = {
        "registration.reg_preview_image1": SimpleNamespace(get_value=lambda: label1),
        "registration.reg_preview_image2": SimpleNamespace(get_value=lambda: label2),
    }
    bare_interface.reg.file_labels = ["image-0", "image-1", "image-2"]
    bare_interface.reg.register_msims = ["msim-0", "msim-1", "msim-2"]
    bare_interface.metrics_methods = []
    bare_interface._preview_overlap_cache = None
    overlap1 = SimpleNamespace(compute=lambda: "overlap1-computed")
    overlap2 = SimpleNamespace(compute=lambda: "overlap2-computed")
    bare_interface.reg.select_pair_overlap.return_value = (overlap1, overlap2, "pixel-space")
    bare_interface.reg.register_overlap.return_value = ("transform", 0.5, {"fixed_points": []})
    monkeypatch.setattr(interface_module, "calc_msims_metrics", lambda *a, **k: {"metrics": True})
    bare_interface.params = {"registration": {"method": "orb"}}


def test_run_preview_registration_reuses_cached_overlap_across_param_changes(
    bare_interface, monkeypatch, mocked_activity_contexts
):
    """The overlap crop depends only on source data and the selected pair/channel - not on
    registration method/tuning - so select_pair_overlap() must run once and register_overlap()
    must reuse the cached crop across parameter-only changes."""
    _stub_preview_registration_deps(bare_interface, monkeypatch)

    bare_interface.params = {"registration": {"method": "orb"}}
    result1 = bare_interface.run_preview_registration()
    bare_interface.params = {"registration": {"method": "sift"}}
    result2 = bare_interface.run_preview_registration()

    assert result1 is not None and result2 is not None
    assert bare_interface.reg.select_pair_overlap.call_count == 1
    assert bare_interface.reg.register_overlap.call_count == 2


def test_run_preview_registration_cache_invalidated_on_pair_change(
    bare_interface, monkeypatch, mocked_activity_contexts
):
    """Selecting a different image pair must invalidate the cached overlap crop."""
    _stub_preview_registration_deps(bare_interface, monkeypatch, label1="image-0", label2="image-1")

    bare_interface.run_preview_registration()

    bare_interface.param_widgets["registration.reg_preview_image2"] = SimpleNamespace(
        get_value=lambda: "image-2"
    )
    bare_interface.run_preview_registration()

    assert bare_interface.reg.select_pair_overlap.call_count == 2


def test_run_preview_registration_cache_invalidated_when_register_msims_changes(
    bare_interface, monkeypatch, mocked_activity_contexts
):
    """preprocess() assigns a new register_msims list object whenever pre-processing actually
    changes something - the cached overlap crop must be invalidated when that identity changes."""
    _stub_preview_registration_deps(bare_interface, monkeypatch)

    bare_interface.run_preview_registration()

    bare_interface.reg.register_msims = ["msim-0-reprocessed", "msim-1-reprocessed", "msim-2-reprocessed"]
    bare_interface.run_preview_registration()

    assert bare_interface.reg.select_pair_overlap.call_count == 2


def test_run_preview_registration_returns_none_on_failure(
    bare_interface, monkeypatch, mocked_activity_contexts
):
    """@catch_run_errors must turn an internal failure into a None return instead of an
    unhandled exception, so preview_registration() can bail out cleanly."""
    _stub_preview_registration_deps(bare_interface, monkeypatch)
    bare_interface.reg.select_pair_overlap.side_effect = ValueError("boom")
    monkeypatch.setattr("muvis_align.ui._utils.show_error", MagicMock())

    result = bare_interface.run_preview_registration()

    assert result is None


@pytest.mark.parametrize(
    ("global_registered", "pairs_registered", "reply", "runs"),
    [
        (True, False, None, False),
        (False, False, "No", False),
        (False, False, "Yes", True),
        (False, True, "Yes", True),
    ],
)
def test_pair_registration_confirmation_paths(
    bare_interface,
    monkeypatch,
    mocked_activity_contexts,
    global_registered,
    pairs_registered,
    reply,
    runs,
):
    bare_interface.viewer = MagicMock()
    bare_interface.reg.is_global_registered.return_value = global_registered
    bare_interface.reg.is_pairs_registered.return_value = pairs_registered
    bare_interface.reg.source_transform_key = "source_metadata"
    bare_interface.run_pair_registration = MagicMock()
    bare_interface.update_registered = MagicMock()
    warning = MagicMock()
    monkeypatch.setattr(interface_module, "show_warning", warning)
    if reply is not None:
        monkeypatch.setattr(
            interface_module.QMessageBox,
            "question",
            lambda *_: getattr(interface_module.QMessageBox, reply),
        )

    bare_interface.pair_registration()

    if global_registered:
        warning.assert_called_once()
    assert bare_interface.run_pair_registration.called is runs
    assert bare_interface.update_registered.called is runs


@pytest.mark.parametrize(
    ("pairs_registered", "reply", "run_pair", "run_global"),
    [
        (False, "No", False, False),
        (False, "Yes", True, True),
        (True, "Yes", False, True),
    ],
)
def test_registration_process_confirmation_and_prerequisites(
    bare_interface,
    monkeypatch,
    mocked_activity_contexts,
    pairs_registered,
    reply,
    run_pair,
    run_global,
):
    bare_interface.viewer = MagicMock()
    bare_interface.params = {"registration": {"operation": "register"}}
    bare_interface.reg.is_global_registered.return_value = False
    bare_interface.reg.is_pairs_registered.return_value = pairs_registered
    bare_interface.reg.msims = ["sim"]
    bare_interface.reg.reg_transform_key = "registered"
    bare_interface.view_msims = ["preview"]
    bare_interface.run_pair_registration = MagicMock()
    bare_interface.run_global_registration = MagicMock()
    bare_interface.enable_tabs = MagicMock()
    bare_interface.update_registered = MagicMock()
    copy = MagicMock()
    monkeypatch.setattr(interface_module, "copy_transforms_to_msims", copy)
    monkeypatch.setattr(
        interface_module.QMessageBox,
        "question",
        lambda *_: getattr(interface_module.QMessageBox, reply),
    )

    bare_interface.registration_process()

    assert bare_interface.run_pair_registration.called is run_pair
    assert bare_interface.run_global_registration.called is run_global
    if run_global:
        # pair and global registration each report their own bar, one after the other - sharing
        # one left the global registration reporting into a bar the pair phases had filled
        bare_interface.run_global_registration.assert_called_once_with()
        if run_pair:
            bare_interface.run_pair_registration.assert_called_once_with()
        copy.assert_called_once_with(["sim"], ["preview"], "registered")
        bare_interface.enable_tabs.assert_called_once_with(True, 4)
        # the refresh is a phase of the registration operation's bar, not a bar of its own
        _, refresh_kwargs = bare_interface.update_registered.call_args
        assert refresh_kwargs['view_transform_key'] == "registered"
        assert refresh_kwargs['progress_factory'] is not None


@pytest.mark.parametrize(
    ("tile_size", "expected"),
    [("1024", 1024), ("512, 1024", [512, 1024])],
)
def test_fusion_process_parses_tile_size_and_updates_state(
    bare_interface,
    monkeypatch,
    mocked_activity_contexts,
    tile_size,
    expected,
):
    bare_interface.viewer = MagicMock()
    bare_interface.params = {
        "registration": {"operation": "register"},
        "input_output": {"registration_dimension": "all"},
        "fusion": {
            "method": "average",
            "spacing": "mean",
            "tile_size": tile_size,
            "ome_version": "0.5",
        },
    }
    bare_interface.reg.is_fused.return_value = False
    bare_interface.reg.fuse.return_value = ("fused", None)
    bare_interface.get_all_widgets = MagicMock(return_value={})
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._napari_view_add_fused_data = MagicMock()
    monkeypatch.setattr(
        interface_module.QMessageBox,
        "question",
        lambda *_: interface_module.QMessageBox.Yes,
    )
    # fuse() always returns real msims in production - this test only cares about tile_size
    # parsing and state transitions, so stand in a passthrough for the msim->sim extraction step
    monkeypatch.setattr(interface_module, "extract_sims_from_fused", lambda result: result)

    bare_interface.fusion_process()

    assert bare_interface.reg.fuse.call_args.kwargs["tile_size"] == expected
    assert (
        bare_interface.reg.fuse.call_args.kwargs["output_filename"]
        == "registered"
    )
    bare_interface._napari_view_add_fused_data.assert_called_once_with(
        bare_interface.viewer, "fused", "Fused"
    )
    assert bare_interface.reg.state is CanonicalRegState.FUSED
    assert bare_interface.view_mode is CanonicalViewMode.FUSED


def test_build_view_msims_uses_native_pyramid_when_available():
    """_build_view_msims() must use a source's own msim as-is when it already has more than one
    native pyramid level - napari can lazily pick whichever resolution it needs from that real
    pyramid, so no downscaling step is needed."""
    from muvis_align.MVSRegistration import MVSRegistration

    reg = MVSRegistration()
    reg.init(
        operation='register',
        input_path=[
            'data/S000/S000_000_000.ome.zarr',
            'data/S000/S000_000_001.ome.zarr',
        ],
        output_path='../../output/test_preview/',
    )
    reg.init_data()
    assert all(len(source.shapes) > 1 for source in reg.sources)  # sanity check: native pyramid

    interface = Interface.__new__(Interface)
    interface.reg = reg

    view_msims = interface._build_view_msims()

    assert view_msims == list(reg.msims)


def test_build_view_msims_downscales_large_single_resolution_source():
    """A single-resolution source (no native pyramid to pick a coarser level from) larger than
    1000px on its largest spatial dimension must be downscaled by one constant factor so that
    dimension becomes ~1000px; a source already at or under 1000px is left untouched."""
    from multiview_stitcher import spatial_image_utils as si_utils
    from muvis_align.image.util import get_msim_image0, wrap_sims_as_msims

    def make_source_and_msim(size):
        sim = si_utils.get_sim_from_array(
            np.zeros((size, size), dtype=np.uint8),
            dims=['y', 'x'],
            scale={'y': 1, 'x': 1},
            translation={'y': 0, 'x': 0},
            transform_key='source_metadata',
        )
        msim = wrap_sims_as_msims([sim])[0]
        source = SimpleNamespace(
            shapes=[(size, size)],
            scale_factors=[{'y': 1.0, 'x': 1.0}],
            get_pixel_size=lambda: {'y': 1.0, 'x': 1.0},
        )
        return source, msim

    large_source, large_msim = make_source_and_msim(2000)
    small_source, small_msim = make_source_and_msim(500)

    interface = Interface.__new__(Interface)
    interface.reg = SimpleNamespace(
        sources=[large_source, small_source],
        msims=[large_msim, small_msim],
        source_transform_key='source_metadata',
    )

    view_msims = interface._build_view_msims()

    large_image0 = get_msim_image0(view_msims[0])
    assert large_image0.sizes['x'] == 1000
    assert large_image0.sizes['y'] == 1000

    small_image0 = get_msim_image0(view_msims[1])
    assert small_image0.sizes['x'] == 500
    assert small_image0.sizes['y'] == 500


def test_create_napari_data_show_preprocessed_handles_fuse_internal_3d_promotion(make_napari_viewer):
    """MVSRegistration.fuse() promotes msims to 3D internally (make_msims_3d) whenever sources
    sit at more than one distinct z position - regardless of whether register_msims (used when
    show_preprocessed=True) already has a 'z' dim, which it doesn't for a plain 'register'
    operation (is_stack=False) with every source individually 2D. _create_napari_data() must
    still hand fuse() an output_chunksize that already accounts for the z dim fuse() is about to
    add, or multiview_stitcher's own chunk-bbox computation crashes with KeyError('z')."""
    from muvis_align.MVSRegistration import MVSRegistration

    source_metadata = {
        'position': {'z': 'fn[-2]', 'y': 0.0, 'x': 'fn[-2]*30'},
        'scale': {'z': '1', 'y': '0.032', 'x': '0.032'},
    }
    reg = MVSRegistration()
    reg.init(
        operation='register',
        input_path=[
            'data/S000/000_000_0.tiff',
            'data/S000/000_001_0.tiff',
        ],
        output_path='../../output/test_fuse_3d_promotion/',
        source_metadata=source_metadata,
    )
    reg.init_data(source_metadata=source_metadata)
    assert len(set(p.get('z') for p in reg.positions)) > 1  # sanity check: genuinely multi-z
    reg.preprocess(reg.msims, scale=None, flatfield_quantiles='', normalisation='none',
                   filter_foreground=False)
    assert 'z' not in reg.register_msims[0]['scale0'].ds['image'].dims  # sanity check: not yet 3D

    interface = Interface.__new__(Interface)
    interface.reg = reg
    interface.params = {'input_output': {'registration_dimension': 'space'}}
    interface.extra_metadata = {}

    fused_msim = interface._create_napari_data(
        reg.source_transform_key, show_preprocessed=True
    )

    assert 'z' in fused_msim['scale0'].ds['image'].dims
    assert fused_msim['scale0'].ds['image'].sizes['z'] == 2


def test_preview_data_layer_is_real_multiscale_pyramid(make_napari_viewer):
    """update_views()'s 'data' preview layer (_create_napari_data -> _napari_view_add_fused_data)
    must show a genuine napari multiscale layer sourced from msims end to end - no sims added to
    the napari image layer - even for the pre-registration preview, not just the post-fusion
    'Fused' export view."""
    from muvis_align.MVSRegistration import MVSRegistration
    from multiview_stitcher import msi_utils

    reg = MVSRegistration()
    reg.init(
        operation='register',
        input_path=[
            'data/S000/S000_000_000.ome.zarr',
            'data/S000/S000_000_001.ome.zarr',
        ],
        output_path='../../output/test_preview/',
    )
    reg.init_data()

    interface = Interface.__new__(Interface)
    interface.reg = reg
    interface.params = {'input_output': {'registration_dimension': 'all'}}
    interface.extra_metadata = {}
    interface.view_msims = interface._build_view_msims()

    fused_msim = interface._create_napari_data(reg.source_transform_key, fusion_method='')
    n_levels = len(msi_utils.get_sorted_scale_keys(fused_msim))
    assert n_levels > 1  # sanity check: fusion produced a real multiscale pyramid, not one level

    viewer = make_napari_viewer()
    interface._napari_view_add_fused_data(viewer, fused_msim, 'data')

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.multiscale is True
    assert len(layer.data) == n_levels
    for level_data, next_level_data in zip(layer.data, layer.data[1:]):
        assert next_level_data.shape[-1] <= level_data.shape[-1]
        assert next_level_data.shape[-2] <= level_data.shape[-2]


def test_napari_view_add_fused_data_shows_real_multiscale_pyramid(make_napari_viewer):
    """fusion_process()'s new full-resolution 'Fused' view (_napari_view_add_fused_data) must
    add a genuine napari multiscale layer - one array per real native pyramid level, not a single
    downsampled preview - since MVSRegistration.fuse() always returns the fused msim (a real
    multiscale pyramid) when called with reg.msims."""
    import yaml
    from muvis_align.MVSRegistration import MVSRegistration
    from multiview_stitcher import msi_utils

    with open(os.path.join('resources', 'params_test_2d.yml'), 'r', encoding='utf8') as file:
        params = yaml.safe_load(file)
    operation_params = params['operations'][0]

    reg = MVSRegistration()
    reg.init_params(params['general'], operation_params)
    reg.init_data()
    reg.preprocess(reg.msims, **operation_params.get('preprocess', {}))
    reg.register(reg.register_msims, reg.register_indices, params=operation_params)

    fused_image, _ = reg.fuse(reg.msims, transform_key=reg.reg_transform_key)
    n_levels = len(msi_utils.get_sorted_scale_keys(fused_image))
    assert n_levels > 1  # sanity check: this run actually produced a real multiscale pyramid

    viewer = make_napari_viewer()
    interface = Interface.__new__(Interface)
    interface.extra_metadata = {}

    interface._napari_view_add_fused_data(viewer, fused_image, 'Fused')

    assert len(viewer.layers) == 1
    layer = viewer.layers[0]
    assert layer.multiscale is True
    assert len(layer.data) == n_levels
    # levels must actually shrink
    for level_data, next_level_data in zip(layer.data, layer.data[1:]):
        assert next_level_data.shape[-1] <= level_data.shape[-1]
        assert next_level_data.shape[-2] <= level_data.shape[-2]


class _RecordingProgressBar:
    """Stands in for a napari progress bar, recording how a phase reported itself."""

    def __init__(self, total=None, desc=None):
        self.total = total
        self.descriptions = [desc]
        self.updates = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def update(self, n=1):
        self.updates += n

    def set_description(self, desc):
        self.descriptions.append(desc)


class _RecordingProgressFactory:
    def __init__(self):
        self.bars = []

    def __call__(self, total=None, desc=None):
        bar = _RecordingProgressBar(total=total, desc=desc)
        self.bars.append(bar)
        return bar


def test_build_view_msims_reports_progress_per_source():
    """Forcing view_msims builds one msim per source - tens of seconds for a few hundred - so it
    must report per-source progress when given a factory, instead of running silently."""
    interface = CanonicalInterface.__new__(CanonicalInterface)
    interface.reg = MagicMock()
    interface.reg.sources = [SimpleNamespace(shapes=[0, 1]), SimpleNamespace(shapes=[0, 1])]
    interface.reg.msims = ['msim-0', 'msim-1']
    factory = _RecordingProgressFactory()

    view_msims = interface._build_view_msims(progress_factory=factory)

    assert view_msims == ['msim-0', 'msim-1']
    assert len(factory.bars) == 1
    bar = factory.bars[0]
    assert bar.total == 2
    assert bar.updates == 2


def test_build_view_msims_without_factory_needs_no_progress():
    interface = CanonicalInterface.__new__(CanonicalInterface)
    interface.reg = MagicMock()
    interface.reg.sources = [SimpleNamespace(shapes=[0, 1])]
    interface.reg.msims = ['msim-0']

    assert interface._build_view_msims() == ['msim-0']


def test_init_progress_reports_saved_project_load(
    bare_interface, monkeypatch, mocked_activity_contexts
):
    """Resuming a saved project forces the per-source msim build and loads the saved
    registration inside reg.init_progress() - the slowest part of a project load - so
    Interface.init_progress() must hand it a progress factory to report that. Drawing the
    view it ends on is a separate operation, and must not report into the load's bar."""
    bare_interface.viewer = MagicMock()
    bare_interface.params = {'registration': {'operation': 'register'}}
    bare_interface.reg.is_pairs_registered.return_value = False
    bare_interface.reg.is_global_registered.return_value = False
    bare_interface.reg.is_fused.return_value = False
    bare_interface.update_views = MagicMock()
    bare_interface.enable_tabs = MagicMock()

    bare_interface.init_progress()

    _, kwargs = bare_interface.reg.init_progress.call_args
    assert callable(kwargs['progress_factory'])
    # the refresh owns its own bar (update_views() opens one when given no factory)
    bare_interface.update_views.assert_called_once_with(show_images=False)


def test_reg_init_progress_builds_msims_with_progress(monkeypatch, tmp_path):
    """MVSRegistration.init_progress() reads self.msims in both of its load branches - that lazy
    build (~15s for 328 sources) must go through ensure_msims(progress_factory=...) so it is
    reported per source, and the whole-set steps that follow it must report themselves too,
    rather than all of it happening silently while a resumed project appears to hang."""
    import networkx as nx
    from muvis_align import MVSRegistration as mvs_module
    from muvis_align.MVSRegistration import MVSRegistration, RegState

    pair_mappings_filename = tmp_path / 'pair_mappings.json'
    pair_mappings_filename.write_text('{}', encoding='utf8')

    reg = MVSRegistration.__new__(MVSRegistration)
    reg.output = str(tmp_path).replace(os.sep, '/') + '/'
    reg.output_params = {}
    reg.state = RegState.PAIRS_REG
    reg.sources = [SimpleNamespace(get_size=lambda: {'y': 10, 'x': 10})]
    reg.filenames = ['file-0']
    reg.source_transform_key = 'source_metadata'
    reg._msims = None
    reg.ensure_msims = MagicMock(return_value=['msim-0'])
    reg.msims = ['msim-0']
    monkeypatch.setattr(mvs_module, 'make_msims_2d', lambda msims: msims)
    monkeypatch.setattr(
        mvs_module.mv_graph, 'build_view_adjacency_graph_from_msims',
        lambda *_, **__: nx.Graph()
    )
    factory = _RecordingProgressFactory()

    reg.init_progress('registered', 'zarr', progress_factory=factory)

    # the later `self.msims` reads hit the cached build (ensure_msims is mocked here, so they
    # still reach it) - what matters is that the first, actually-building call carries the factory
    assert reg.ensure_msims.call_args_list[0] == call(progress_factory=factory)
    # the msim build reports itself per source (inside ensure_msims, mocked here); the two
    # whole-set steps left in this branch report as named steps of one bar
    assert len(factory.bars) == 1
    bar = factory.bars[0]
    assert bar.total == 2
    assert bar.updates == 2


class _FakeNapariProgress:
    """Stands in for napari's progress bar, recording everything a shared bar would show."""

    instances = []

    def __init__(self, total=None, desc=None, **kwargs):
        self.total = total
        self.descriptions = [desc]
        self.value = 0
        self.closed = False
        self.kwargs = kwargs
        _FakeNapariProgress.instances.append(self)

    def update(self, n=1):
        self.value += n

    def set_description(self, desc):
        self.descriptions.append(desc)

    def close(self):
        self.closed = True


def test_phase_progress_fills_one_bar_once_across_phases():
    """Every phase of one operation reports into a single bar that fills once: each phase moves
    it across its own slice, so the bar never restarts. (Phases that instead added to the bar's
    total made 2/2 become 2/330, which reads as a new bar starting, not as progress.)"""
    from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

    _FakeNapariProgress.instances = []
    with NapariPhaseProgress(progress_class=_FakeNapariProgress, desc='Loading project',
                             phases=2) as factory:
        with factory(total=3, desc='Building sources') as pbar:
            for _ in range(3):
                pbar.update(1)
            after_first_phase = _FakeNapariProgress.instances[0].value
        with factory(total=2, desc='Loading pair registration') as pbar:
            pbar.update(1)
            pbar.set_description('Building pair graph')
            pbar.update(1)

    assert len(_FakeNapariProgress.instances) == 1
    bar = _FakeNapariProgress.instances[0]
    # the total is fixed for the whole operation, and the first of two phases leaves the bar
    # half full rather than full
    assert bar.total == NapariPhaseProgress.ticks
    assert after_first_phase == NapariPhaseProgress.ticks // 2
    assert bar.value == NapariPhaseProgress.ticks
    # one description for the whole operation - phases naming themselves would turn one bar
    # into a flicker of labels
    assert bar.descriptions == ['Loading project']
    assert bar.closed


def test_phase_progress_never_moves_backwards():
    """Whatever the phases report, the bar only advances - an unexpected extra phase takes what
    is left rather than resizing the bar under the user."""
    from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

    _FakeNapariProgress.instances = []
    values = []
    with NapariPhaseProgress(progress_class=_FakeNapariProgress, phases=2) as factory:
        for total, desc in [(4, 'Building sources'), (300, 'Building views'), (2, 'Extra')]:
            with factory(total=total, desc=desc) as pbar:
                for _ in range(total):
                    pbar.update(1)
                    values.append(_FakeNapariProgress.instances[0].value)

    assert values == sorted(values)
    assert _FakeNapariProgress.instances[0].value == NapariPhaseProgress.ticks


def test_phase_progress_shows_no_bar_when_nothing_reports():
    """An operation that turns out to have nothing to report must not flash an empty bar."""
    from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

    _FakeNapariProgress.instances = []
    with NapariPhaseProgress(progress_class=_FakeNapariProgress, desc='Loading project'):
        pass

    assert _FakeNapariProgress.instances == []


def test_phase_progress_completes_a_phase_that_undercounts():
    """A phase that declares steps it does not spend (a whole-set step counted as one) must not
    leave the shared bar permanently short of its own total."""
    from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

    _FakeNapariProgress.instances = []
    with NapariPhaseProgress(progress_class=_FakeNapariProgress) as factory:
        with factory(total=4, desc='Building views'):
            pass

    bar = _FakeNapariProgress.instances[0]
    assert (bar.value, bar.total) == (NapariPhaseProgress.ticks, NapariPhaseProgress.ticks)


def test_update_views_reports_into_the_callers_bar(bare_interface, monkeypatch):
    """A refresh that is part of a larger operation (loading a saved project) must report as
    phases of that operation's bar, and leave the activity dock / widget state to it."""
    from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

    bare_interface.viewer = MagicMock()
    bare_interface.overview = MagicMock()
    bare_interface.reg.sources = [SimpleNamespace(get_size=lambda: {"y": 10, "x": 10})]
    bare_interface.reg.positions = [{"z": 0}]
    bare_interface._create_napari_shapes = MagicMock(return_value=([], [], [], []))
    bare_interface._clear_napari_view = MagicMock()
    bare_interface._update_view_add_shapes = MagicMock()
    dock = MagicMock()
    monkeypatch.setattr(interface_module, "VisibleActivityDock", dock)

    _FakeNapariProgress.instances = []
    with NapariPhaseProgress(progress_class=_FakeNapariProgress, desc='Loading project') as factory:
        with factory(total=1, desc='Loading pair registration') as pbar:
            pbar.update(1)
        bare_interface.update_views(transform_key="source_metadata", show_images=False,
                                    progress_factory=factory)

    dock.assert_not_called()
    # one bar, still the caller's - the refresh opened none of its own, and did not rename it
    assert len(_FakeNapariProgress.instances) == 1
    bar = _FakeNapariProgress.instances[0]
    assert bar.total == NapariPhaseProgress.ticks
    assert bar.descriptions == ['Loading project']


def test_phase_progress_tqdm_class_reports_into_the_same_bar():
    """A library's own tqdm loop (multiview_stitcher's fusion, patched in by NapariMVSProgress)
    must land on the operation's bar as one more phase, not open a bar beside it."""
    from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

    _FakeNapariProgress.instances = []
    with NapariPhaseProgress(progress_class=_FakeNapariProgress, desc='Fusion',
                             phases=2) as factory:
        with factory(total=1, desc='Preparing fusion') as pbar:
            pbar.update(1)
        for _ in factory.tqdm_class(range(3), desc='Fusing blocks'):
            pass

    assert len(_FakeNapariProgress.instances) == 1
    bar = _FakeNapariProgress.instances[0]
    assert (bar.value, bar.total) == (NapariPhaseProgress.ticks, NapariPhaseProgress.ticks)
    assert bar.descriptions == ['Fusion']


def test_phase_progress_tqdm_class_tolerates_the_rest_of_tqdm():
    """tqdm's surface is wide and a library may touch any of it mid-loop - the stand-in must no-op
    rather than raise from inside someone else's fusion."""
    from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

    _FakeNapariProgress.instances = []
    with NapariPhaseProgress(progress_class=_FakeNapariProgress) as factory:
        pbar = factory.tqdm_class(total=2, desc='Fusing blocks')
        pbar.set_postfix(loss=1)
        pbar.refresh()
        pbar.update(2)
        pbar.close()

    assert _FakeNapariProgress.instances[0].value == NapariPhaseProgress.ticks


def test_pre_processing_process_reports_work_and_view_separately(bare_interface):
    """Pre-processing and the view refresh it triggers are two operations, in that order: the
    work reports its own bar, then building and drawing the view reports a second one - neither
    is handed the other's factory."""
    bare_interface.viewer = MagicMock()
    bare_interface.run_pre_processing = MagicMock(return_value=True)
    bare_interface.update_views = MagicMock()
    bare_interface.enable_tabs = MagicMock()
    bare_interface.enable_modify_pair_registration = MagicMock()
    bare_interface.select_tab = MagicMock()

    bare_interface.pre_processing_process()

    bare_interface.run_pre_processing.assert_called_once_with()
    bare_interface.update_views.assert_called_once_with(show_preprocessed=True)


def test_only_one_operation_bar_at_a_time(bare_interface, monkeypatch):
    """Two bars must never be on screen together: an operation started while another is running
    reports into the running one, and leaves it the activity dock and the widget state."""
    dock = MagicMock()
    widgets = MagicMock()
    monkeypatch.setattr(interface_module, "VisibleActivityDock", dock)
    monkeypatch.setattr(interface_module, "TemporarilyDisabledWidgets", widgets)
    bare_interface.viewer = MagicMock()

    with bare_interface._operation_progress('Initialising sources') as outer:
        with bare_interface._operation_progress('Refreshing view') as inner:
            assert inner is outer

    dock.assert_called_once_with(bare_interface.viewer)
    widgets.assert_called_once_with(bare_interface.enable_plugin_widget)
    # ...and the next operation, once this one has finished, opens a bar of its own again
    assert bare_interface._running_operation is None


def test_phase_progress_leaves_room_for_work_after_its_phases():
    """A phase must never fill the bar: the operation is still running, and an unexpected extra
    phase (another dask compute, a second registration) still needs somewhere to go. Only the
    end of the operation fills it."""
    from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

    _FakeNapariProgress.instances = []
    with NapariPhaseProgress(progress_class=_FakeNapariProgress, phases=1) as factory:
        with factory(total=2, desc='Declared phase') as pbar:
            pbar.update(1)
            pbar.update(1)
        after_declared_phases = _FakeNapariProgress.instances[0].value
        with factory(total=2, desc='Unexpected extra phase') as pbar:
            pbar.update(1)
            pbar.update(1)
        after_extra_phase = _FakeNapariProgress.instances[0].value

    bar = _FakeNapariProgress.instances[0]
    assert after_declared_phases < NapariPhaseProgress.ticks
    assert after_extra_phase > after_declared_phases
    assert after_extra_phase < NapariPhaseProgress.ticks
    assert bar.value == NapariPhaseProgress.ticks  # ...and finishing the operation fills it


def test_update_metadata_source_refresh_reports_its_own_bar(bare_interface, monkeypatch):
    """The refresh at the end of update_metadata_source() must open its own bar. It used to be
    handed the source-initialisation factory, which by then belonged to a finished operation -
    so re-running input/output showed no bar for the refresh at all."""
    bare_interface.viewer = MagicMock()
    bare_interface.reg.is_pairs_registered.return_value = False
    bare_interface.reg.is_initialised.return_value = True
    bare_interface.reg.source_transform_key = 'source_metadata'
    bare_interface.source_metadata = {}
    bare_interface.update_views = MagicMock()
    for name in ['populate_channels', 'populate_coordinate_systems', 'populate_channels_table',
                 'populate_metadata_table', 'check_3d_view']:
        setattr(bare_interface, name, MagicMock())
    bare_interface.update_output_channels = MagicMock(return_value=False)

    assert bare_interface.update_metadata_source() is True

    bare_interface.update_views.assert_called_once_with(show_images=False)


def test_phase_progress_reused_after_its_operation_opens_a_new_bar():
    """A factory that outlives its operation (handed on to work that runs after it) must report
    on a new bar - reporting on the closed one would be reporting nowhere."""
    from muvis_align.ui.NapariPhaseProgress import NapariPhaseProgress

    _FakeNapariProgress.instances = []
    factory = NapariPhaseProgress(progress_class=_FakeNapariProgress, desc='Initialising sources')
    with factory:
        with factory(total=2) as pbar:
            pbar.update(2)
    with factory(total=2) as pbar:
        pbar.update(2)

    assert len(_FakeNapariProgress.instances) == 2
    first, second = _FakeNapariProgress.instances
    assert first.closed
    assert second.value > 0
