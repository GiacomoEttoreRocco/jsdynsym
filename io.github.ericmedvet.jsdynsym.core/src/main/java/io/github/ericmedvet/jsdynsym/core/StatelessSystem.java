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

package io.github.ericmedvet.jsdynsym.core;

/**
 * @author "Eric Medvet" on 2023/02/25 for jsdynsym
 */
@FunctionalInterface
public interface StatelessSystem<I, O> extends DynamicalSystem<I, O, StatelessSystem.State> {

  record State() {
    public static State EMPTY = new State();
  }

  @Override
  default State getState() {
    return State.EMPTY;
  }

  @Override
  default void reset() {
  }
}
