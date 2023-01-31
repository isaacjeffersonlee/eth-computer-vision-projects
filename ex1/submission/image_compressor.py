# Note: You are not allowed to import additional python packages except NumPy
import numpy as np

# TODO: Cleanup code
# TODO: Upload to Google colab and upload results


def pad_image(
    x_i: np.ndarray, kernel_size: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    pad_vert = kernel_size[0] // 2
    pad_horiz = kernel_size[1] // 2
    return (
        np.pad(
            array=x_i,
            pad_width=(pad_vert, pad_horiz),
            mode="constant",
            constant_values=(255, 255),
        ),
        (pad_vert, pad_horiz),
    )


def de_noise(
    X_i: np.ndarray, image_height: int = 96, image_width: int = 96
) -> np.ndarray:
    """Filter out noise using median filtering."""
    h, w = image_height, image_width

    def de_noise_x_i(x_i: np.ndarray) -> np.ndarray:
        """Remove noise from an individual image, x_i."""
        x_i_de_noised = x_i.copy()
        kernel_size = (3, 3)
        x_i_padded, pad_width = pad_image(x_i, kernel_size)
        h_i, h_j = pad_width
        # Iterate from left to right, row by row
        for i_0 in range(x_i.shape[0]):
            for j_0 in range(x_i.shape[1]):
                i = i_0 + h_i
                j = j_0 + h_j
                window = x_i_padded[i - h_i : i + h_i + 1, j - h_j : j + h_j + 1]
                x_i_de_noised[i_0, j_0] = np.median(window)

        return x_i_de_noised

    if len(X_i.shape) == 2:  # Multiple images
        N = X_i.shape[1]
        X_i = np.reshape(X_i.T, (N, h, w))  # Convert to 2D images
        X_i_de_noised = []
        for x_i in X_i:
            X_i_de_noised.append(de_noise_x_i(x_i))
        # Re-flatten individual images
        X_i_de_noised = np.reshape(np.array(X_i_de_noised), (N, h * w)).T

    elif len(X_i.shape) == 1:  # Single image
        x_i = np.reshape(X_i, (h, w))  # Convert to 2D image
        x_i_de_noised = de_noise_x_i(x_i)
        X_i_de_noised = np.reshape(
            x_i_de_noised, (h * w,)
        )  # Re-flatten individual images

    else:
        raise NotImplementedError(
            f"de_noise() is not implemented for arrays of shape: {X_i.shape}"
        )
    return X_i_de_noised


class ImageCompressor:
    # This class is responsible to i) learn the codebook given the training images
    # and ii) compress an input image using the learnt codebook.
    def __init__(self, remove_noise: bool = True):
        self.remove_noise = remove_noise
        self.mean_image = np.array([])
        # TODO: Ask about whether we need principal components?
        # self.principal_components = np.array([])
        self.principal_directions = np.array([])
        self.codebook = np.array([])
        self.train_images = np.array([])  # Gets set by train method
        self.N = None
        self.image_height = None
        self.image_width = None

    def get_2D_channel_matrices(self, images: np.ndarray) -> dict[np.ndarray]:
        """Separate the (R,G,B) image into separate colour channels.

        Convert 4D: N x h x w x 3 np array into three 2D: h*w x N np arrays.
        Or 3D: w x h x 3 np array into three 2D: w x h arrays.
        Representing the matrices for each Red, Green, Blue colour channels,
        with the ith column of the matrix representing the r or g or b pixel
        values of the ith image in the N samples.
        """
        X = np.array(images)
        X_dict = {}
        if len(X.shape) == 4:
            # Split into channels
            for n, color_str in enumerate(["r", "g", "b"]):
                X_i = X.flat[n::3].reshape((X.shape[0], X.shape[1], X.shape[2]))
                X_dict[color_str] = np.transpose(
                    X_i.reshape((X_i.shape[0], X_i.shape[1] * X_i.shape[2]))
                )
        elif len(X.shape) == 3:
            # Split into channels
            for n, color_str in enumerate(["r", "g", "b"]):
                X_dict[color_str] = X.flat[n::3].reshape((X.shape[0], X.shape[1]))
        else:
            raise NotImplementedError(
                f"get_2D_channel does not support matrices of shape: {X.shape}"
            )

        return X_dict

    def get_mean_image(self, X_i: np.ndarray) -> np.ndarray:
        return np.transpose(np.average(np.transpose(X_i), axis=0))

    def get_principals(self, X_i: np.ndarray) -> tuple[np.ndarray]:
        """Get the principal direction matrix and principal components"""
        U, S, V_T = np.linalg.svd(X_i, full_matrices=False)
        return (U, S**2)

    def get_codebook(self) -> np.ndarray:
        """Return all information needed for compression as a single np array.

        If codebook has not yet been learned,
        then call the train function to learn it."""
        if self.codebook.size == 0:
            self.train(self.train_images)

        return self.codebook

    def de_mean(self, X_i: np.ndarray, X_i_mean: np.ndarray) -> np.ndarray:
        """De-mean X_i and return result."""
        return np.transpose(np.transpose(X_i) - np.transpose(X_i_mean))

    def train(self, train_images: list[np.ndarray], k: int = 15) -> None:
        """Given a list of training images, learn the codebook."""
        # Update class attributes using train_images
        self.train_images = np.array(train_images)
        self.N = self.train_images.shape[0]
        self.image_height = self.train_images.shape[1]
        self.image_width = self.train_images.shape[2]
        # k == number of principal components
        assert k < 100, "Maximum dimension cannot be larger than original sample!"
        mean_image = []
        # principal_components = []
        principal_directions = []
        # Split images into their respective colour channels
        X_dict = self.get_2D_channel_matrices(train_images)

        for X_i, color_str in zip(X_dict.values(), X_dict.keys()):
            if self.remove_noise:
                X_i = de_noise(
                    X_i=X_i,
                    image_height=self.image_height,
                    image_width=self.image_width,
                )
            X_i_mean = self.get_mean_image(X_i)
            X_i_0 = self.de_mean(X_i, X_i_mean)
            U_i, S_i = self.get_principals(X_i_0)
            mean_image.append(X_i_mean)
            # principal_components.append(S_i)
            principal_directions.append(U_i[:, :k])

        self.mean_image = np.array(mean_image)
        # self.principal_components = np.array(principal_components)
        self.principal_directions = np.array(principal_directions)

        self.codebook = np.array(
            [
                np.hstack(
                    (
                        np.array([self.mean_image[i].astype(np.float16)]).T,
                        self.principal_directions[i].astype(np.float16),
                    )
                )
                for i in range(3)
            ]
        )

    def compress(self, test_image: np.ndarray) -> np.ndarray:
        """De-noise and reduce the dimensions of test_image using the learned codebook."""
        # Separate into channels
        x_dict = self.get_2D_channel_matrices(test_image)
        compressed_image = []
        for i, x_i in enumerate(x_dict.values()):
            U_i = self.codebook[i][:, 1:]
            x_i = x_i.flatten()
            x_i = de_noise(x_i)
            x_i = x_i - self.mean_image[i]
            z_i = np.matmul(U_i.T, x_i)
            compressed_image.append(z_i)

        return np.array(compressed_image).astype(np.float16)


class ImageReconstructor:
    # This class is used on the client side to reconstruct the compressed images.
    def __init__(self, codebook: np.ndarray, remove_noise: bool = False):
        # The codebook learnt by the ImageCompressor is passed as input when
        # initializing the ImageReconstructor
        self.remove_noise = remove_noise
        mean_image = []
        principle_directions = []
        for i in range(3):
            mean_image.append(codebook[i][:, 0])
            U_i = codebook[i][:, 1:]
            principle_directions.append(U_i)

        self.mean_image = np.array(mean_image)
        self.principal_directions = np.array(principle_directions)

    def reconstruct(self, test_image_compressed):
        # Given a compressed test image, this function should reconstruct the original image
        X = []
        for i in range(3):
            U_i = self.principal_directions[i]
            z_i = test_image_compressed[i]
            mu_i = self.mean_image[i]
            x_i = np.matmul(U_i, z_i) + mu_i
            if self.remove_noise:
                X.append(de_noise(x_i))
            else:
                X.append(x_i)

        return np.array(X).T.reshape((96, 96, 3)).astype(int)


if __name__ == "__main__":
    im = ImageCompressor()
