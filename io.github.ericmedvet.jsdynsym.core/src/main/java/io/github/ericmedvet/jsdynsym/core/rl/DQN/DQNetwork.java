package io.github.ericmedvet.jsdynsym.core.rl.DQN;
import io.github.ericmedvet.jsdynsym.core.rl.QLearning;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;

public class DQNetwork {
    int hiddenNeurons = 128;
    public DQNetwork(int nInputs, int nOutputs) {
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(123).weightInit(WeightInit.XAVIER);
        MultiLayerConfiguration configuration = builder.list()
                .layer(new DenseLayer.Builder().nIn(nInputs).nOut(hiddenNeurons).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nIn(hiddenNeurons).nOut(hiddenNeurons).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(hiddenNeurons).nOut(nOutputs).activation(Activation.IDENTITY).build()).build();
        this.network = new MultiLayerNetwork(configuration);
        this.network.init();
    }
    private MultiLayerNetwork network;

    public MultiLayerNetwork getNetwork() {
        return network;
    }

    public INDArray params(){
        return network.params();
    }

    public void setParams(INDArray params){
        network.setParams(params);
    }

    public double[] forward(double[] state) {
        INDArray input = Nd4j.create(state);
        INDArray output = network.output(input);
        return output.toDoubleVector();
    }

    public List<double[]> predict(List<double[]> states) {
        List<double[]> predictions = new ArrayList<>();
        for (double[] state : states) {
            predictions.add(forward(state));
        }
        return predictions;
    }

    public void fit(double[] states, double[] targets) {
        INDArray input = Nd4j.create(states);
        INDArray target = Nd4j.create(targets);
        network.fit(input, target);
    }

    public void softUpdate(DQNetwork policyNetwork){
        INDArray policyNetParams = policyNetwork.params(); // Parametri del modello policyNet
        INDArray targetNetParams = this.params(); // Parametri del modello targetNet
        double tau = 0.005; // Valore di tau per il soft update
        targetNetParams.muli(1 - tau);
        targetNetParams.addi(policyNetParams.mul(tau));
        this.setParams(targetNetParams); // Impostazione dei nuovi parametri di targetNet
    }

}
