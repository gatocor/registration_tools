import ultrack

class Tracking():
    def __init__(self):
        return
    
class TrackingUltrack(Tracking):
    def __init__(self):
        super().__init__()
        
        self._cfg = ultrack.MainConfig()
        self._tracks_df = None
        self._graph = None

        return

    def load(self, path):

        self._cfg = ultrack.load_config(path)

        return

    def fit(self, detection, edges, scale=None):

        if scale is None:
            try:
                voxel_size = detection.attrs["scale"]
            except:
                raise ValueError("Scale not provided and not found in mask attributes.")

        ultrack.track(
            self._cfg,
            detection=detection,
            edges=edges,
            scale=voxel_size,
            overwrite=True,
        )

        self._tracks_df, self._graph = ultrack.to_tracks_layer(self._cfg)

        return