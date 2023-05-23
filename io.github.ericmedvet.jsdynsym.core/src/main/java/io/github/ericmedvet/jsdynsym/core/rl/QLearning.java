/*
 * Copyright 2023 eric
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.ericmedvet.jsdynsym.core.rl;

import java.util.*;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class QLearning implements EnumeratedTimeInvariantReinforcementLearningAgent<QLearning.State> {
    private final int nOfInputs;
    private final int nOfOutputs;
    private final double explorationRate;
    private final double learningRate;
    private final double discountFactor;
    private final RandomGenerator randomGenerator;
    private final State state;
    private ObservationActionPair previousPair;

    public QLearning(int nOfInputs, int nOfOutputs, double learningRate, double discountFactor, double explorationRate, RandomGenerator randomGenerator) {
        this.nOfInputs = nOfInputs;
        this.nOfOutputs = nOfOutputs;
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.explorationRate = explorationRate;
        this.randomGenerator = randomGenerator;
        state = new State(nOfInputs, nOfOutputs, new HashMap<>());
    }

    public record ObservationActionPair(int observation, int action) {
    }

    public record State(int nOfInputs, int nOfOutputs, Map<ObservationActionPair, Double> table) {
        @Override
        public String toString() {
            return IntStream.range(0, nOfOutputs)
                    .mapToObj(o -> IntStream.range(0, nOfInputs)
                            .mapToObj(i -> "%5.2f".formatted(table().getOrDefault(new ObservationActionPair(i, o), Double.NaN)))
                            .collect(Collectors.joining(" "))
                    )
                    .collect(Collectors.joining("\n"));
        }
    }

    @Override
    public State getState() {
        return state;
    }

    @Override
    public void reset() {
        state.table().clear();
    }

    @Override
    public int nOfInputs() {
        return nOfInputs;
    }

    @Override
    public int nOfOutputs() {
        return nOfOutputs;
    }

    public Integer greedyAction(Integer input) {
        int output;
        List<Map.Entry<ObservationActionPair, Double>> oEntries = state.table().entrySet().stream().filter(e -> e.getKey().observation() == input).toList();
        Optional<Map.Entry<ObservationActionPair, Double>> oEntry = oEntries.stream().max(Map.Entry.comparingByValue());
        if (oEntry.isEmpty()) {
            output = randomGenerator.nextInt(nOfOutputs);
        } else {
            double value = oEntry.get().getValue();
            if (value > 0 || oEntries.size() == nOfOutputs) {
                //the best action has a value greater than the default one (0)
                output = oEntry.get().getKey().action();
            } else {
                //choose a random action among the one never chosen
                output = preferNeverChosenAction(oEntries);
            }
        }
        return output;
    }

    public Integer explorationAction(Integer input) {
        int output;
        List<Map.Entry<ObservationActionPair, Double>> oEntries = state.table().entrySet().stream().filter(e -> e.getKey().observation() == input).toList();
        //Optional<Map.Entry<ObservationActionPair, Double>> oEntry = oEntries.stream().max(Map.Entry.comparingByValue()); Since we are exploring, we don't need to find the best action
        if (oEntries.size() == nOfOutputs) {
            //choose random action
            output = randomGenerator.nextInt(nOfOutputs);
        } else {
            //choose a random action among the one never chosen
            output = preferNeverChosenAction(oEntries);
        }
        return output;
    }

    private int preferNeverChosenAction(List<Map.Entry<ObservationActionPair, Double>> oEntries) {
        List<Integer> chosenActions = oEntries.stream().map(e -> e.getKey().action()).toList();
        List<Integer> allActions = IntStream.range(0, nOfOutputs).boxed().toList();
        List<Integer> availableActions = new ArrayList<>(allActions);
        availableActions.removeAll(chosenActions);
        return availableActions.get(randomGenerator.nextInt(availableActions.size()));
    }

    public void updateTable(ObservationActionPair entry, double reward, Integer nextObservation) {
        double oldValue = state.table().getOrDefault(entry, 0.0);
        double newValue = oldValue + learningRate * (reward + discountFactor *
                state.table().getOrDefault(new ObservationActionPair(nextObservation, greedyAction(nextObservation)), 0.0) - oldValue);
        state.table().put(entry, newValue);
        //System.out.println("Old value: " + oldValue + " New value: " + newValue);
    }

    @Override
    public Integer step(Integer input, double previousReward) {
        if (previousPair != null) {
            updateTable(previousPair, previousReward, input);
        }
        int output;
        if (randomGenerator.nextDouble() < explorationRate) {
            output = explorationAction(input);
        } else {
            output = greedyAction(input);
        }
        previousPair = new ObservationActionPair(input, output);
        return output;
    }

}
