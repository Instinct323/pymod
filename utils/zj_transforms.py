import numpy as np

__all__ = ["TFmanager"]


class _TransformManager(dict):
    """
    A transformation manager that keeps track of transformations between different frames.

    The transformations are stored in a dictionary with the following structure:
    {src: {dst1: T_src_dst1, dst2: T_src_dst2, ...}, ...}
    """

    def __repr__(self):
        return "TFmanager:\n" + "\n".join(f"\t{src} -> {', '.join(dsts)}" for src, dsts in self.items())

    def register(self,
                 T_src_dst: np.ndarray,
                 src: str,
                 dst: str):
        """
        Updates the internal transformation graph by registering both direct and indirect
        transformations based on transitivity.

        :param T_src_dst: 4x4 transformation matrix from `src` to `dst`.
        :param src: Name of the source frame.
        :param dst: Name of the destination (parent) frame.
        """
        assert src not in self, f"Frame '{src}' already registered."
        assert T_src_dst.shape == (4, 4), "Transformation matrix must be 4x4."
        self.setdefault(src, {})
        self.setdefault(dst, {})
        # Store the direct transformation from src to dst
        self[src][dst] = T_src_dst
        # Propagate transformations from dst's children to src
        for frame, T_dst_frame in self[dst].copy().items():
            self[src][frame] = T_dst_frame @ T_src_dst
            self[frame][src] = np.linalg.inv(self[src][frame])
        # Store inverse transformation from dst to src
        self[dst][src] = np.linalg.inv(T_src_dst)

    def apply_vec(self,
                  vec: np.ndarray,
                  src: str,
                  dst: str) -> np.ndarray:
        """ Apply rotation only to vectors. """
        tf = self[src, dst]
        return np.einsum('ij, ...j -> ...i', tf[:3, :3], vec)

    def apply_pcd(self,
                  pcd: np.ndarray,
                  src: str,
                  dst: str) -> np.ndarray:
        """ Apply full transformation (rotation + translation) to a point cloud. """
        tf = self[src, dst]
        if pcd.shape[-1] == 4:
            return np.einsum('ij, ...j -> ...i', tf, pcd)
        elif pcd.shape[-1] == 3:
            return np.einsum('ij, ...j -> ...i', tf[:3, :3], pcd) + tf[:3, 3]


TFmanager = _TransformManager()

if __name__ == '__main__':
    def randomT():
        T = np.eye(4)
        T[:3] = np.random.rand(3, 4)
        return T


    # Register some example transformations
    TFmanager.register(randomT(), "camera", "world")
    TFmanager.register(randomT(), "robot", "camera")
    print(TFmanager)
    print(TFmanager.register.__doc__)

    pcd = np.random.rand(300, 100, 4)

    # Test point cloud transformation
    res1 = TFmanager.apply_pcd(pcd, "camera", "world")
    res2 = (TFmanager["camera", "world"] @ pcd.reshape(-1, 4).T).T.reshape(*pcd.shape)
    print(np.abs(res1 - res2).sum())

    # Test vector transformation
    res3 = TFmanager.apply_vec(pcd[..., :3], "camera", "world")
    res4 = (TFmanager["camera", "world"][:3, :3] @ pcd[..., :3].reshape(-1, 3).T).T.reshape(*pcd[..., :3].shape)
    print(np.abs(res3 - res4).sum())
