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

    public void optimize_model() {
        if (replayMemory.size() < batchSize) {
            return;
        }
        List<Transition> batch = replayMemory.sample(batchSize);
        List<Double[]> states = new ArrayList<>(batchSize);
        List<Integer> actions = new ArrayList<>(batchSize);
        List<Double> rewards = new ArrayList<>(batchSize);
        List<Double[]> nextStates = new ArrayList<>(batchSize);
        for (Transition transition : batch) {
            states.add(transition.state());
            actions.add(transition.action());
            rewards.add(transition.reward());
            nextStates.add(transition.nextState());
        }
        List<Double[]> qValues = targetNetwork.predict(states); // restituisce una lista dove per ogni stato hai un array di valori Q per ogni azione
        List<Double[]> nextQValues = policyNetwork.predict(nextStates);
        List<Double> expectedStateActionValues = new ArrayList<>(nextQValues.stream().map(i -> Collections.max(Arrays.asList(i)) * this.gamma).toList());
        for (int i = 0; i < expectedStateActionValues.size(); i++) {
            expectedStateActionValues.set(i, expectedStateActionValues.get(i) + rewards.get(i));
        }
        List<Double> statesActionValues = IntStream.of(actions.size()).mapToObj(i -> qValues.get(i)[actions.get(i)]).toList();
        List<Double> loss = IntStream.of(statesActionValues.size()).mapToObj(i -> Math.pow(expectedStateActionValues.get(i) - statesActionValues.get(i), 2)).toList();
        policyNetwork.fit(statesActionValues, expectedStateActionValues);
    }
    public int selectAction(Double[] state) {
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
    public int greedyAction(Double[] state) {
        Double[] qValues = this.policyNetwork.forward(state);
        return Arrays.asList(qValues).indexOf(Collections.max(Arrays.asList(qValues)));
    }
    public void saveInMemory(Double[] state, int action, double reward, Double[] nextState){
        this.replayMemory.push(new Transition(state, action, reward, nextState));
    }
    Double[] previousState;
    int previousAction;
    public Integer step(Double[] observation, Double previousReward) {
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
