package io.github.ericmedvet.jsdynsym.core.rl.DQN;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ReplayMemory {
    private final int capacity;
    private final List<Transition> memory;
    private int index;
    public ReplayMemory(int capacity) {
        this.capacity = capacity;
        this.memory = new ArrayList<Transition>(capacity);
        this.index = 0;
    }
    public void push(Transition transition) {
        if (memory.size() < capacity) {
            memory.add(transition);
        } else {
            // Overwrite oldest transition
            memory.set(index % capacity, transition);
            index++;
        }
    }
    public List<Transition> sample(int batchSize) {
        List<Transition> batch = new ArrayList<>(batchSize);
        Random random = new Random();

        for (int i = 0; i < batchSize; i++) {
            int randomIndex = random.nextInt(memory.size());
            batch.add(memory.get(randomIndex));
        }

        return batch;
    }
    public int size() {
        return memory.size();
    }
}


