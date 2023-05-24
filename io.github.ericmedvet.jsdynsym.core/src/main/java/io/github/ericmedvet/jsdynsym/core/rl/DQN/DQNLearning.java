package io.github.ericmedvet.jsdynsym.core.rl.DQN;

import io.github.ericmedvet.jsdynsym.core.rl.QLearning;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.stream.IntStream;
public class DQNLearning {
    private final DQNetwork targetNetwork;
    private final DQNetwork policyNetwork;
    private final ReplayMemory replayMemory;
    private final int batchSize;
    private final double gamma;
    private final double epsilon;
    private final int nOutputs;
    private final int nInputs;
    public DQNLearning(int nInputs, int nOutputs, ReplayMemory replayMemory, int batchSize, double gamma, double epsilon) {
        this.nInputs = nInputs;
        this.nOutputs = nOutputs;
        this.targetNetwork = new DQNetwork(nInputs, nOutputs);
        this.policyNetwork = new DQNetwork(nInputs, nOutputs);
        this.replayMemory = replayMemory;
        this.batchSize = batchSize;
        this.gamma = gamma;
        this.epsilon = epsilon;
    }
    public int getnOutputs() {
        return nOutputs;
    }
    public int getnInputs() {
        return nInputs;
    }
    public void optimize_model() {
        if (replayMemory.size() < batchSize) {
            return;
        }
        List<Transition> batch = replayMemory.sample(batchSize);
        List<double[]> states = new ArrayList<>(batchSize);
        int[] actions = new int[batchSize];
        double[] rewards = new double[batchSize];
        List<double[]> nextStates = new ArrayList<>(batchSize);
        Transition transition;
        for(int i = 0; i< batchSize; i++){
            transition = batch.get(i);
            states.add(batch.get(i).state());
            actions[i] = transition.action();
            rewards[i] = transition.reward();
            nextStates.add(transition.nextState());
        }
        List<double[]> qValues = targetNetwork.predict(states);
        List<double[]> nextQValues = policyNetwork.predict(nextStates);
        double[] expectedStateActionValues = new double[batchSize];
        double[] statesActionValues = new double[batchSize];
        for (int i = 0; i < batchSize; i++) {
            expectedStateActionValues[i] = rewards[i] + gamma * Arrays.stream(nextQValues.get(i)).max().getAsDouble();
            statesActionValues[i] = qValues.get(i)[actions[i]];
        }

        policyNetwork.fit(statesActionValues, expectedStateActionValues);
    }
    public int selectAction(double[] state) {
        double sample = Math.random();
        int action;
        if (sample < this.epsilon) {
            action = this.explorationAction();
        } else {
            action = this.greedyAction(state);
        }
        return action;
    }
    public int explorationAction() {
        Random random = new Random();
        return random.nextInt(this.nOutputs);
    }
    public int greedyAction(double[] state) {
        double[] qValues = this.policyNetwork.forward(state);
        //return Arrays.asList(qValues).indexOf(Collections.max(Arrays.asList(qValues)));
        int maxIndex = -1;
        double maxValue = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < qValues.length; i++) {
            if (qValues[i] > maxValue) {
                maxValue = qValues[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
    public void saveInMemory(double[] state, int action, double reward, double[] nextState){
        this.replayMemory.push(new Transition(state, action, reward, nextState));
    }
    double[] previousState;
    int previousAction;
    public Integer step(double[] observation, Double previousReward) {
        if (previousReward != null) {
            this.saveInMemory(previousState, previousAction, previousReward, observation);
        }
        int output;
        output = this.selectAction(observation);

        previousState = observation;
        previousAction = output;

        this.optimize_model();
        this.targetNetwork.softUpdate(this.policyNetwork);

        return output;
    }
}
