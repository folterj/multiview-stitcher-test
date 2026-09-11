import logging
import threading
import time


def _paint_now():
    """Give Qt one pass to paint what was just shown.

    Everything here runs on the Qt thread, so a bar put up immediately before a long blocking
    call would not actually appear until that call returned - which is the one time it is worth
    having.
    """
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:  # pragma: no cover - Qt is always present in the napari plugin
        return
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


class NapariPhaseProgress:
    """One napari progress bar, filling once, for a whole user-facing operation.

    Used as the progress_factory the phases already expect (see MVSRegistration._build_msims(),
    preprocess(), init_progress(), Interface._build_view_msims()): a phase asks for a bar of its
    own with `factory(total=..., desc=...)` and gets a slice of this one instead. The bar runs
    from empty to full exactly once per operation - a phase moves it across its own slice only,
    so it never restarts and never jumps backwards. Without this, phases sharing a bar by adding
    to its total each looked like a new bar starting: 2/2 becoming 2/330 reads as a reset, not
    as progress.

    `phases` is how many phases the operation expects, which is what sizes the slices; an
    operation that turns out to run more gets them out of what is left, and one that runs fewer
    has the remainder filled in when it ends.

    The bar goes up when the operation starts, not when its first phase reports - an operation
    whose first stretch of work reports nothing (global registration spends most of itself
    inside one blocking call) would otherwise show nothing at all until it was nearly done -
    and the Qt event loop is pumped once so it is actually painted before that work begins.

    `emit` makes a headless twin of one of these, for work running off the Qt thread (see
    Interface._run_off_thread()): it drives no bar of its own and instead reports the position
    it would have moved one to, for the bar on the Qt thread to follow. Qt widgets may only be
    touched from the thread that owns them, so a worker never holds the real bar.

    A bar is no use to a headless run, so a long operation also says where it has got to in
    the log, every `heartbeat_seconds` - which is the only way to tell a slow phase (a preview
    fusion over a few thousand sources takes many minutes to plan) from a hung one.

    The bar keeps the operation's own description throughout - a phase naming itself would turn
    one bar into a flicker of labels, and the phases are internal steps of the operation, not
    something to follow. The tick count is internal too: the default bar_format keeps the
    elapsed/remaining estimate but leaves the raw counts out of what napari displays.
    """

    # what napari shows beside the bar: no counts, no rate, just the time estimate (its eta
    # label is everything after the last '|' of tqdm's formatted line)
    bar_format = '{desc}|{elapsed}<{remaining}'

    # the bar counts in ticks of the whole operation, not in any phase's own units - a phase
    # maps its own steps onto its slice of these
    ticks = 1000

    # how often a running operation says where it has got to in the log. Long enough that a
    # short operation never logs at all, short enough to see that a long one is alive
    heartbeat_seconds = 30

    # what the last phase the operation expects may take of what is left, so that a phase can
    # never fill the bar: only the end of the operation does that (a full bar then always means
    # finished, not 'the declared phases are done but it is still working'), and an unexpected
    # extra phase - another dask compute, another registration - always has somewhere to go
    last_phase_share = 0.9

    def __init__(self, progress_class=None, desc=None, phases=1, min_duration=0.0, emit=None,
                 **progress_kwargs):
        from napari.utils import progress

        self.progress_class = progress_class or progress
        self.emit = emit
        self.desc = desc
        progress_kwargs.setdefault('bar_format', self.bar_format)
        self.phases = max(int(phases), 1)
        self.phases_left = self.phases
        self.min_duration = max(float(min_duration), 0.0)
        self.progress_kwargs = progress_kwargs
        self._pbar = None
        self._started_at = None
        self._position = 0.0
        self._target = 0.0
        self._moving = False
        self._heartbeat = None
        self._done = threading.Event()

    def __enter__(self):
        self._started_at = time.monotonic()
        if self.emit is None:
            kwargs = dict(self.progress_kwargs)
            kwargs['total'] = self.ticks
            if self.desc is not None:
                kwargs['desc'] = self.desc
            self._pbar = self.progress_class(**kwargs)
            _paint_now()
        self._start_heartbeat()
        return self

    def __call__(self, total=None, desc=None, **_):
        return _ProgressPhase(self, total, desc)

    def _start_heartbeat(self):
        if self.emit is not None or not self.heartbeat_seconds:
            # the headless twin of an operation reports through the one that owns the bar, which
            # is the one doing the logging
            return
        self._done.clear()

        def heartbeat():
            while not self._done.wait(self.heartbeat_seconds):
                elapsed = time.monotonic() - (self._started_at or time.monotonic())
                so_far = (f'{elapsed / 60:.1f} minutes' if elapsed >= 60
                          else f'{elapsed:.0f} seconds')
                logging.info(f'{self.desc or "Working"}:'
                             f' {self._position / self.ticks * 100:.0f}%'
                             f' ({so_far} so far)')

        self._heartbeat = threading.Thread(target=heartbeat, daemon=True,
                                           name='muvis-align progress')
        self._heartbeat.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self._done.set()
        self._heartbeat = None
        if self._pbar is not None:
            if exc_type is None:
                self._move_to(self.ticks)
            self._pbar.close()
            self._pbar = None
        # back to how this started: a factory that outlives its operation (handed on to work
        # that runs after it) then reports on a new bar, rather than silently on a closed one
        self.phases_left = self.phases
        self._position = 0.0
        self._target = 0.0
        if self._started_at is not None and self.min_duration > 0:
            # an operation that turned out to be instant would otherwise flash the activity
            # dock open and shut
            wait_s = self.min_duration - (time.monotonic() - self._started_at)
            if wait_s > 0:
                time.sleep(wait_s)
        self._started_at = None
        return False

    @property
    def tqdm_class(self):
        """A tqdm stand-in whose bars are phases of this one.

        Lets the progress reporting inside a third-party library (the fusion loop in
        multiview_stitcher, patched in by NapariMVSProgress) move this operation's bar along
        as one more phase, instead of opening a bar of its own next to it.
        """
        owner = self

        class _PhaseTqdm(_ProgressPhase):
            def __init__(self, iterable=None, desc=None, total=None, **kwargs):
                if total is None:
                    try:
                        total = len(iterable)
                    except TypeError:
                        total = None
                super().__init__(owner, total=total, desc=desc)
                self.iterable = iterable
                self._open = False

            def __iter__(self):
                self._start()
                for item in self.iterable:
                    yield item
                    self.update(1)
                self.close()

            def _start(self):
                if not self._open:
                    self.__enter__()
                    self._open = True

            def update(self, n=1):
                self._start()
                super().update(n)

            def close(self):
                if self._open:
                    self._open = False
                    self.__exit__(None, None, None)

            def __getattr__(self, name):
                # tqdm has a wide surface (set_postfix, refresh, write, clear, ...) that a
                # library may touch anywhere in its loop - anything beyond the small bar
                # interface above is a no-op here rather than an AttributeError raised in the
                # middle of someone else's fusion
                return lambda *args, **kwargs: None

        return _PhaseTqdm

    def _begin_phase(self, total, desc=None):
        remaining = self.ticks - self._position
        if self.phases_left > 1:
            span = remaining / self.phases_left
        else:
            # the last phase the operation expects, or one it never expected at all, takes most
            # of what is left rather than all of it
            span = remaining * self.last_phase_share
        self.phases_left = max(self.phases_left - 1, 0)
        return _PhaseSlice(start=self._position, span=span, total=total)

    def _advance_phase(self, phase_slice, n=1):
        phase_slice.done += n
        if phase_slice.total:
            fraction = min(phase_slice.done / phase_slice.total, 1.0)
        else:
            # a phase that never said how many steps it has can still say it is running: show it
            # half way through its own slice until it ends
            fraction = 0.5
        self._move_to(phase_slice.start + phase_slice.span * fraction)

    def _end_phase(self, phase_slice):
        self._move_to(phase_slice.start + phase_slice.span)

    def set_position(self, position):
        """Move the bar to a position reported from elsewhere - by the headless twin driving
        it from a worker thread (see Interface._run_off_thread())."""
        self._move_to(position)

    def _move_to(self, position):
        # Never re-entered, however it is called. Updating a napari bar repaints it, which pumps
        # the Qt event loop (QtLabeledProgressBar.setValue), which can deliver the next position
        # a worker has reported - straight back into here. Left to recurse, a load reporting
        # hundreds of positions runs the stack out and takes the process with it, so a move
        # already in progress just raises the target it is heading for.
        self._target = min(max(position, self._position, self._target), self.ticks)
        if self._moving:
            return
        self._moving = True
        try:
            while True:
                # the bar counts in whole ticks, so only a move that crosses one shows up
                step = int(self._target) - int(self._position)
                self._position = self._target
                if step <= 0:
                    break
                if self.emit is not None:
                    self.emit(int(self._position))
                elif self._pbar is not None:
                    self._pbar.update(step)
                if self._target <= self._position:
                    break
        finally:
            self._moving = False


class _PhaseSlice:
    """What one phase was given of the operation: where its slice starts, how much of the bar it
    spans, and how many of the phase's own steps that span is worth."""

    def __init__(self, start, span, total):
        self.start = start
        self.span = span
        self.total = total
        self.done = 0


class _ProgressPhase:
    """One phase's view of the operation's bar - the same (context manager, update,
    set_description) interface a napari progress bar offers, so phases need no knowledge of the
    sharing."""

    def __init__(self, owner, total=None, desc=None):
        self.owner = owner
        self.total = total
        self.desc = desc
        self._slice = None

    def __enter__(self):
        self._slice = self.owner._begin_phase(self.total, self.desc)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None and self._slice is not None:
            self.owner._end_phase(self._slice)
        return False

    def update(self, n=1):
        if self._slice is not None:
            self.owner._advance_phase(self._slice, n)

    def set_description(self, desc):
        # accepted (phases and third-party bars call it) but ignored: the bar shows the
        # operation's description for as long as the operation runs
        pass
