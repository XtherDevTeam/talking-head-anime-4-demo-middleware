## Animation Configuration File Specification

This document details the complete specifications for creating `*.json` animation configuration files for the THA4-based real-time animation renderer. The system is designed to translate a high-level, declarative JSON file into a smooth, continuous sequence of character animations.

### 1. High-Level Overview

The system is built around three main concepts:

*   **`AnimationConfiguration`**: The top-level object that loads and holds all animation rules from a single JSON file.
*   **States**: The character can be in various states, such as `idle`, `talking`, or `surprised`. Each state has a unique set of animations associated with it. The `idle` state is special and runs in a continuous, randomized loop.
*   **Compositions**: An animation is built by composing one or more actions together (e.g., an eye wink and an eyebrow raise). These compositions are defined with parameters like duration, target values, and transition styles.

---

### 2. JSON File Structure

The configuration is a single JSON object with the following top-level keys:

```json
{
  "name": "MyCharacter",
  "description": "Animation configuration for MyCharacter.",
  "states": { ... },
  "breathing": { ... }
}
```

#### **Top-Level Keys**

| Key           | Type     | Required | Description                                                                                                                                                             |
| :------------ | :------- | :------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`        | `string` | No       | An identifier for the configuration. Defaults to "Unnamed Animation Configuration".                                                                                    |
| `description` | `string` | No       | A human-readable description.                                                                                                                                           |
| `states`      | `object` | Yes      | A dictionary containing all defined animation states for the character.                                                                                                 |
| `breathing`   | `object` | No       | An optional configuration for a continuous, global breathing animation that is overlaid on all other states. It is highly recommended for creating a natural, living feel. |

#### **2.1 The `states` Object**

The `states` object maps state names to a list of possible animations for that state.

```json
"states": {
  "idle": [
    // Array of Animation Groups for the idle state
  ],
  "thinking": [
    // Array of Animation Groups for the "thinking" state
  ]
}
```

*   **`idle` State (Required):** The `idle` state is mandatory. When the renderer is in this state, it will randomly select one **Animation Group** from the `idle` array, play it, and then randomly select another. This creates a continuous, non-repetitive idle loop.
*   **Custom States (Optional):** You can define any number of other states (e.g., `"talking"`, `"surprised"`). These animations will only be played when the renderer's state is explicitly changed to that key name.

#### **2.2 The `breathing` Object**

If present, this object defines a continuous sinusoidal breathing motion.

```json
"breathing": {
  "duration": 4.0
}
```

| Key        | Type    | Required | Description                                                    |
| :--------- | :------ | :------- | :------------------------------------------------------------- |
| `duration` | `float` | Yes      | The time in seconds for one full breath cycle (inhale + exhale). |

---

### 3. Animation Group and Composition

The core of any state is the **Animation Group**, which is an array of **Animation Compositions**.

*   **Animation Group** (`array<AnimationComposition>`): A list of compositions. **All compositions within a group are played concurrently.** The engine calculates the total duration required to play every composition in the group (including offsets and restore times) and generates a single, unified animation sequence.

*   **Animation Composition** (`object`): A single, atomic animation instruction that controls one part of the character.

#### **3.1 Animation Composition Parameters**

Here is the specification for a single composition object.

```json
{
  "action": "eye",
  "desired_state": "eye_wink",
  "duration": 0.15,
  "x": [0.0, 1.0],
  "y": [0.0, 1.0],
  "kick_off_offset": 0.5,
  "restore": "reverse",
  "transition": "linear"
}
```

| Key               | Type                         | Required                               | Description                                                                                                                                                                  |
| :---------------- | :--------------------------- | :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `action`          | `string`                     | Yes                                    | The parameter to animate. See Section 4 for a full list of available actions.                                                                                                |
| `duration`        | `float`                      | Yes                                    | The time in seconds for the forward part of the animation (from start value to end value).                                                                                   |
| `x`, `y`, `z`     | `array<float, float>`        | Yes (depends on `action`)              | A two-element array `[start, end]` defining the motion. Values are normalized from `0.0` to `1.0`. See Section 4 for which parameters each action uses.                |
| `desired_state`   | `string`                     | Yes (for `eyebrow`, `eye`)             | The specific named morph to use for the animation. See Section 4 for available options.                                                                                      |
| `shapes`          | `array<string>`              | Yes (for `mouth`)                      | An array of mouth shape names to cycle through during the animation's duration.                                                                                              |
| `kick_off_offset` | `float`                      | No (Default: `0.0`)                    | A delay in seconds before this composition starts playing within its group. This allows for sequencing actions within a concurrent group.                                        |
| `restore`         | `string`                     | No (Default: `"reverse"`)              | Defines how the parameter returns to its default state after `duration` is complete. <br> - `"reverse"`: Plays the animation backward over the same duration. <br> - `"rapid"`: Plays the animation backward at double speed (half the duration). <br> - `"none"`: Instantly snaps back to the default state. |
| `transition`      | `string`                     | No (Default: `"linear"`)               | The interpolation method used to move between the start and end values. Currently, only `"linear"` and `"sine"` is supported.                                                            |

---

### 4. Master Parameter and Options Reference

This table provides the ground truth for all available `action` types, their required parameters, their valid options, and the normalized range for their values.

| `action` Value      | Parameters Used in JSON | Parameter Mapping to Model | Available `desired_state` / `shapes` Options                                                                                                                                                 | Parameter Value Range |
| :------------------ | :---------------------- | :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------- |
| `eyebrow`           | `x`, `y`                | Left Slider, Right Slider  | `eyebrow_troubled`, `eyebrow_angry`, `eyebrow_lowered`, `eyebrow_raised`, `eyebrow_happy`, `eyebrow_serious`                                                                                   | `[0.0, 1.0]`          |
| `eye`               | `x`, `y`                | Left Slider, Right Slider  | `eye_wink`, `eye_happy_wink`, `eye_surprised`, `eye_relaxed`, `eye_unimpressed`, `eye_raised_lower_eyelid`                                                                                   | `[0.0, 1.0]`          |
| `mouth`             | `shapes` (array)        | Active Mouth Shape         | `mouth_aaa`, `mouth_iii`, `mouth_uuu`, `mouth_eee`, `mouth_ooo`, `mouth_delta`, `mouth_lowered_corner`, `mouth_raised_corner`, `mouth_smirk`                                                 | N/A (List of strings) |
| `iris_small`        | `x`, `y`                | Left Slider, Right Slider  | N/A                                                                                                                                                                                          | `[0.0, 1.0]`          |
| `iris_rotation`     | `x`, `y`                | `iris_rotation_x`, `iris_rotation_y` | N/A                                                                                                                                                                                          | `[0.0, 1.0]`          |
| `head_rotation`     | `x`, `y`, `z`           | `head_x`, `head_y`, `neck_z` | N/A                                                                                                                                                                                          | `[0.0, 1.0]`          |
| `body_rotation`     | `y`, `z`                | `body_y`, `body_z`         | N/A                                                                                                                                                                                          | `[0.0, 1.0]`          |

**Note on Parameter Ranges:** All numerical parameter values (`x`, `y`, `z`) are **normalized to a `[0.0, 1.0]` range**.
*   For bidirectional parameters like rotations (which the model sees as `[-1.0, 1.0]`), a value of `0.0` corresponds to `-1.0`, `0.5` corresponds to `0.0` (center), and `1.0` corresponds to `1.0`.
*   For unidirectional parameters like `breathing` (model range `[0.0, 1.0]`), the mapping is direct.