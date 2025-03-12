# Quality metrics

### Metrics returned by multiview-sticher register()
- Quality: Pairwise spearman correlation on the overlap between tiles
- Residual: the distance by which the groupwise resolution shifts pairs of views with respect to their pairwise registered positions
  - Virtual point correspondences between pairs of views, defined by the pairwise registration parameters. the residuals are the distances between these points after groupwise resolution. How are the points defined? It's basically the vertices of the overlap area/volume for each of the views. The method is similar to the one used in bigstitcher (section 12): https://www.janelia.org/sites/default/files/H%C3%B6rl%202019.SOM_.pdf
