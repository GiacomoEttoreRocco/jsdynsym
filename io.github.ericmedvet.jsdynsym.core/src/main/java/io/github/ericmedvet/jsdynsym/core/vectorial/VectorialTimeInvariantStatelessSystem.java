package io.github.ericmedvet.jsdynsym.core.vectorial;

import io.github.ericmedvet.jsdynsym.core.StatelessSystem;
import io.github.ericmedvet.jsdynsym.core.TimeInvariantStatelessSystem;

/**
 * @author "Eric Medvet" on 2023/02/25 for jsdynsym
 */
public interface VectorialTimeInvariantStatelessSystem extends VectorialDynamicalSystem<StatelessSystem.State>, TimeInvariantStatelessSystem<double[], double[]> {
}
