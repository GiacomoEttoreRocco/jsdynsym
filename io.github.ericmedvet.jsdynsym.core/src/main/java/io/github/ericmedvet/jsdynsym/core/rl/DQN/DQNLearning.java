package io.github.ericmedvet.jsdynsym.core.rl.DQN;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

public class DQNLearning {
    private final DQNetwork network;
    private final ReplayMemory replayMemory;
    private final int batchSize;
    private final double gamma;
    private final double epsilon;
    private final double epsilonDecay;
    private final double epsilonMin;
    private final int targetUpdate;
    private int step;
    private double epsilonCurrent;

    public DQNLearning(DQNetwork network, ReplayMemory replayMemory, int batchSize, double gamma, double epsilon, double epsilonDecay, double epsilonMin, int targetUpdate) {
        this.network = network;
        this.replayMemory = replayMemory;
        this.batchSize = batchSize;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
        this.targetUpdate = targetUpdate;
        this.step = 0;
        this.epsilonCurrent = epsilon;
    }

    public void learn() {
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

        List<Double[]> qValues = network.predict(states);
        List<Double[]> nextQValues = network.predict(nextStates);

        List<Double> expectedStateActionValues = new ArrayList<>(nextQValues.stream().map(i -> Collections.max(Arrays.asList(i)) * this.gamma).toList());

        for (int i = 0; i < expectedStateActionValues.size(); i++) {
            expectedStateActionValues.set(i, expectedStateActionValues.get(i) + rewards.get(i));
        }

        List<Double> stateAction = IntStream.of(actions.size()).mapToObj(i -> states.get(i)[actions.get(i)]).toList();

        // ...

    }
}
