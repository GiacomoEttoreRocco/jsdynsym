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

public interface NumericalReinforcementLearningAgent<S> extends ReinforcementLearningAgent<double[], double[], S> {

  int nOfInputs();

  int nOfOutputs();

  default void checkDimension(int nOfInputs, int nOfOutputs) {
    if (nOfInputs() != nOfInputs) {
      throw new IllegalArgumentException("Wrong number of inputs: %d found, %d expected".formatted(
          nOfInputs(),
          nOfInputs
      ));
    }
    if (nOfOutputs() != nOfOutputs) {
      throw new IllegalArgumentException("Wrong number of outputs: %d found, %d expected".formatted(
          nOfOutputs(),
          nOfOutputs
      ));
    }
  }
}
