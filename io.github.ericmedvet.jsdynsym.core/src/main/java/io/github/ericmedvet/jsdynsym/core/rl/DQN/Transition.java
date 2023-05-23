package io.github.ericmedvet.jsdynsym.core.rl.DQN;

public record Transition(Double[] state, int action, double reward, Double[] nextState) {}
