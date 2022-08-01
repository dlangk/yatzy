# Overview of Yatzy algorithms

The following file gives you and overview of datamodel, files and methods for this Yatzy system.

## Core Concepts

The game is built around the following core concepts:

* **GameEnvironment**. The GameEnvironment defines all the operations that can be applied to the GameState. It also
  contains a list of the combinations of dices allowed.
    * **Die checking**. The data model and methods used to check if a set of die can be used to score on a certain row
      is organized by a map. In this map are all scoring rows (e.g. fours, one pair etc). For each scoring row we
      identify a validator, a scoring method, potentially a die value and potentially a die count.
        * **Validators**. Validators check if a set of die fulfill conditions for a certain scoring row.
        * **Scoring methods**. Scoring methods assume that a set of die fulfill conditions for a certain scoring row,
          and determines the score.
    * **Applying actions to update state**. The GameEnvironment is also responsible for executing PlayerActions on the
      GameState. A player can decide to take action, and the engine then applies that to the state, thereby updating the
      state according to the action selected. This also includes methods to identify legal combinations (i.e. available
      scoring rows), ensure proposed actions are legal etc. Finally, it also takes care of initializing the GameState,
      and identifying when the game is over.