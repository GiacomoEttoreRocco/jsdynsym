package io.github.ericmedvet.jsdynsym.core.rl.DQN;

public record Transition(double[] state, int action, double reward, double[] nextState) {}
