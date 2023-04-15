import os
import sys
import numpy as np

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from process_models import ImageClassifier, SpikeInput, OutputProcess

sys.path.append('.')


def main() -> None:
    num_images = 25
    num_steps_per_image = 128

    # Create Process instances
    spike_input = SpikeInput(
        vth=1, num_images=num_images, num_steps_per_image=num_steps_per_image
    )
    mnist_clf = ImageClassifier(
        trained_weights_path=os.path.join(".", "mnist_pretrained.npy")
    )
    output_proc = OutputProcess(num_images=num_images)

    # Connect Processes
    spike_input.spikes_out.connect(mnist_clf.spikes_in)
    mnist_clf.spikes_out.connect(output_proc.spikes_in)
    # Connect Input directly to Output for ground truth labels
    spike_input.label_out.connect(output_proc.label_in)

    # Loop over all images
    for img_id in range(num_images):
        print(f"\rCurrent image: {img_id+1}", end="")

        # Run each image-inference for fixed number of steps
        mnist_clf.run(
            condition=RunSteps(num_steps=num_steps_per_image),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True, select_tag="fixed_pt"),
        )

        # Reset internal neural state of LIF neurons
        mnist_clf.lif1_u.set(np.zeros((64,), dtype=np.int32))
        mnist_clf.lif1_v.set(np.zeros((64,), dtype=np.int32))
        mnist_clf.lif2_u.set(np.zeros((64,), dtype=np.int32))
        mnist_clf.lif2_v.set(np.zeros((64,), dtype=np.int32))
        mnist_clf.oplif_u.set(np.zeros((10,), dtype=np.int32))
        mnist_clf.oplif_v.set(np.zeros((10,), dtype=np.int32))

    # Gather ground truth and predictions before stopping exec
    ground_truth = output_proc.gt_labels.get().astype(np.int32)
    predictions = output_proc.pred_labels.get().astype(np.int32)

    # Stop the execution
    mnist_clf.stop()

    accuracy = np.sum(ground_truth == predictions) / ground_truth.size * 100

    print(
        f"\nGround truth: {ground_truth}\n"
        f"Predictions : {predictions}\n"
        f"Accuracy    : {accuracy}"
    )


if __name__ == "__main__":
    main()
