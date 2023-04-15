# Import parent classes for ProcessModels
from lava.magma.core.model.sub.model import AbstractSubProcessModel

# Import execution protocol and hardware resources
from lava.magma.core.resources import CPU

# Import decorators
from lava.magma.core.decorator import implements, requires

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense

from procesess.image_process import ImageClassifier


@implements(ImageClassifier)
@requires(CPU)
class PyImageClassifierModel(AbstractSubProcessModel):
    def __init__(self, proc):
        self.dense0 = Dense(weights=proc.w_dense0.init)
        self.lif1 = LIF(shape=(64,), bias_mant=proc.b_lif1.init, vth=400, dv=0, du=4095)
        self.dense1 = Dense(weights=proc.w_dense1.init)
        self.lif2 = LIF(shape=(64,), bias_mant=proc.b_lif2.init, vth=350, dv=0, du=4095)
        self.dense2 = Dense(weights=proc.w_dense2.init)
        self.output_lif = LIF(
            shape=(10,), bias_mant=proc.b_output_lif.init, vth=1, dv=0, du=4095
        )

        proc.spikes_in.connect(self.dense0.s_in)
        self.dense0.a_out.connect(self.lif1.a_in)
        self.lif1.s_out.connect(self.dense1.s_in)
        self.dense1.a_out.connect(self.lif2.a_in)
        self.lif2.s_out.connect(self.dense2.s_in)
        self.dense2.a_out.connect(self.output_lif.a_in)
        self.output_lif.s_out.connect(proc.spikes_out)

        # Create aliases of SubProcess variables
        proc.lif1_u.alias(self.lif1.u)
        proc.lif1_v.alias(self.lif1.v)
        proc.lif2_u.alias(self.lif2.u)
        proc.lif2_v.alias(self.lif2.v)
        proc.oplif_u.alias(self.output_lif.u)
        proc.oplif_v.alias(self.output_lif.v)
