package io.github.ericmedvet.jsdynsym.core.rl.DQN;

import io.github.ericmedvet.jsdynsym.core.rl.QLearning;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

public class DQNetwork {
    public DQNetwork(int nInputs, int nOutputs) {

    }

    public INDArray params(){
        return null;
    }

    public void setParams(INDArray params){

    }

    public Double[] forward(Double[] state) {
        return new Double[0];
    }

    public List<Double[]> predict(List<Double[]> states) {
        return new ArrayList<Double[]>();
    }

    public void fit(List<Double> states, List<Double> targets) {
        return;
    }

    public void softUpdate(DQNetwork policyNetwork){
        INDArray policyNetParams = policyNetwork.params(); // Parametri del modello policyNet
        INDArray targetNetParams = this.params(); // Parametri del modello targetNet

        double tau = 0.001; // Valore di tau per il soft update

        targetNetParams.muli(1 - tau);
        targetNetParams.addi(policyNetParams.mul(tau));

        this.setParams(targetNetParams); // Impostazione dei nuovi parametri di targetNet
    }

}
