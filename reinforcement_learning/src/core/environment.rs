// reinforcement_learning/src/core/environment.rs

/// The foundational contract for all simulation environments.
/// 
/// Using associated types (`Observation` and `Action`) ensures that your solution 
/// algorithms remain completely decoupled from the underlying environment implementation 
/// (e.g., discrete gridworlds vs. continuous tensor spaces).
pub trait Environment {
    /// The type representing the state or observation space (e.g., `usize`, `ndarray::Array1<f64>`)
    type Observation;
    
    /// The type representing the action space (e.g., `usize` for discrete actions, `ndarray::Array1<f64>` for continuous)
    type Action;

    /// Resets the environment to an initial state and returns the initial observation.
    fn reset(&mut self) -> Self::Observation;

    /// Steps the environment forward by one time step using the given action.
    /// 
    /// Returns a tuple containing:
    /// 1. `Self::Observation` - The next state observation.
    /// 2. `f64` - The reward scalar obtained from the transition.
    /// 3. `bool` - The `terminated` flag (e.g., reaching a terminal goal or falling in a hole).
    /// 4. `bool` - The `truncated` flag (e.g., hitting a maximum episode step limit).
    fn step(&mut self, action: Self::Action) -> (Self::Observation, f64, bool, bool);
}
