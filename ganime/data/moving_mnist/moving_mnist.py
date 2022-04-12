"""moving_mnist dataset."""
import numpy as np
import tensorflow_datasets as tfds

# TODO(moving_mnist): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(moving_mnist): BibTeX citation
_CITATION = """
"""


class MovingMnist(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for moving_mnist dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(moving_mnist): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "video": tfds.features.Video(shape=(20, 64, 64, 1)),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # ("video", "reconstructed"),  # Set to `None` to disable
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(moving_mnist): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
        )

        # TODO(moving_mnist): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        data = np.load(path)
        data = np.moveaxis(data, 0, 1)
        # Also expand dimensions to have channels at the end (n_samples, window, width, height, channels)
        data = np.expand_dims(data, axis=-1)
        # TODO(moving_mnist): Yields (key, example) tuples from the dataset
        for i, video in enumerate(data):
            yield i, {"video": video}
