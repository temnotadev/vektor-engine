# SCR language reference

This project uses a tiny scene-description scripting language stored in `.scr` files such as [scene.scr](./scene.scr). The engine parses the file line by line and turns it into a runtime scene graph plus event handlers. It is not a general-purpose programming language; it is a declarative, imperative scene DSL for objects, lights, camera motion, and scripted interactions.

The implementation lives in [world.hpp](./world.hpp), and the runtime uses the script by loading the file on startup and reloading it when the user presses F5.

## 1. File model

A `.scr` file is a list of commands, one per line.

- `#` starts a comment only when it appears at the beginning of a line or after whitespace.
- `//` works the same way.
- Blank lines are ignored.
- Commands outside an event block run once when the script is loaded.
- Event blocks begin with `on ...` and must close with `end`.

The parser builds an `Event` list, where each event contains a block of script lines and metadata such as key, interval, or collision pair.

## 2. Top-level commands

The engine recognizes the following commands at script scope.

### 2.1 Asset loading

```scr
texture <id> <image>
model   <id> <file.obj> [raw]
```

- `texture` loads a PNG/JPG/TGA/BMP image and registers it under an ID.
- `model` loads an OBJ mesh and registers it by name.
- `raw` skips the model auto-fit and uses the OBJ coordinates directly.
- Asset paths are resolved relative to the script file location first, then relative to configured asset directories.

### 2.2 Scene creation and destruction

```scr
spawn <name> <cube|quad|sphere|modelId> [pos x y z] [rot x y z] [size x y z | scale s]
      [color r g b] [tex id] [uvscale s] [emit r g b] [collide on|off]

destroy <name | prefix*>
set     <name> pos|rot|size|color|emit|dir ...
move    <name> dx dy dz
rotate  <name> dx dy dz
collider <name> [pos x y z] [rot x y z] [size x y z]
```

- `spawn` creates an object from a built-in primitive or a loaded model.
- `destroy` deletes an object or all objects whose names begin with a prefix ending in `*`.
- `set` changes object properties or light properties after creation.
- `move` translates an object or light by a vector.
- `rotate` changes an object rotation in degrees.

Built-in mesh IDs include:

- `cube`
- `quad`
- `sphere`
- any model ID introduced by `model`

A name ending in `#` is expanded automatically into a numbered instance during creation. For example:

```scr
spawn crate# cube pos 0 0 0 size 0.5 0.5 0.5
```

becomes `crate1`, `crate2`, `crate3`, and so on as the script runs.

### 2.3 Lights

```scr
light point <name> x y z r g b range [radius]
light spot  <name> x y z dx dy dz r g b range outerDeg innerDeg [radius]
light sun   <name> dx dy dz r g b [angularRadiusDeg]
```

Lights are stored as separate `LightDef` objects in the world.

- `point`: omnidirectional light with a position and range.
- `spot`: directional cone light with a direction, angles, and range.
- `sun`: directional light, represented like a distant light source.

### 2.4 Camera

```scr
camera pos x y z
camera look yawDeg pitchDeg
camera lock on|off
```

- `camera pos` moves the camera.
- `camera look` sets yaw and pitch in degrees.
- `camera lock` freezes camera movement when enabled.

### 2.5 Variables and arithmetic

```scr
var $v value
calc $v a + - * / min max b
rand $v min max
```

Variables are stored as floating-point values keyed by names that begin with `$`.

Examples:

```scr
var $speed 2.5
calc $distance $speed * 10
rand $x -2 2
```

The implementation supports these arithmetic operators:

- `+`
- `-`
- `*`
- `/`
- `min`
- `max`

A value can be either a literal number or another variable reference such as `$camx`.

### 2.6 Print and control flow

```scr
print <text>
if a < > <= >= == != b
  ...
endif
```

- `print` writes text to stdout.
- Variables are printed as their numeric value when used as `$name`.
- `if` evaluates a comparison and executes the following block until `endif`.

The script engine supports the comparisons:

- `<`
- `>`
- `<=`
- `>=`
- `==`
- `!=`

### 2.7 Built-in world setup

```scr
builtin cornell
builtin pointlights [N]
builtin spots
builtin sun
builtin player [height] [gravity on|off]
```

These commands generate common scene layouts:

- `cornell`: a Cornell box with walls and sample boxes
- `pointlights [N]`: creates a set of point lights
- `spots`: creates spot lights
- `sun`: creates a sun-like directional light
- `player`: enables first-person WASD/mouse movement with jumping and collision

`builtin player [height] [gravity on|off]` enables the first-person player controller: WASD movement, mouse look, gravity, Space to jump, and solid-object collision. Shift moves faster. The optional height is in world units, defaults to `1.8`, and must be greater than twice the controller radius (`0.3`). The camera height scales with it. Set `gravity off` to disable falling and use Q/E to fly vertically; collision remains enabled. The controller starts at the current camera position and automatically enables player collision.

`collider` adds an invisible, solid box. Its center, rotation, and size use the same world units as `spawn`; omitted values default to position/rotation zero and size one. It can be moved, resized, rotated, toggled, or removed with the usual object commands.

Invisible collision boxes can be assigned in a scene file independently of rendered geometry:

```scr
collider hallway_wall pos 0 1.5 -4 size 8 3 0.2
collider ramp_block pos 2 0.5 1 rot 0 20 0 size 2 1 3
builtin player 1.3 gravity off
```

The player is a capsule-like body approximated by three vertical collision spheres; colliders and other solid objects use oriented boxes. Collision-only boxes are omitted from the rendered scene, but are visible and editable as wireframes in SCR Editor and round-trip through its script importer/exporter.

### 2.8 Global render options

```scr
area on|off
gi path|cache|screen
gi rate 1|2
gi cellsize v
denoise on|off|<value>
collision camera on|off [radius]
collision <name|prefix*> on|off
```

These commands configure renderer state:

- `area`: toggles emissive triangles as light sources.
- `gi`: chooses path tracing, surface cache, or screen-space assist modes.
- `gi rate`: switches the GI trace rate.
- `gi cellsize`: controls cache texel cell size.
- `denoise`: enables NRD denoising when the value is positive; `0` disables it.
- `collision camera`: blocks camera movement through solid objects.
- `collision <name>`: toggles object solidity for collisions.

## 3. Event system

The script supports these event blocks:

```scr
on start
  ...
end

on key F
  ...
end

on every 3
  ...
end

on frame
  ...
end

on collide crate* ball
  ...
end
```

### 3.1 `on start`

Runs once when the script starts. This is used for initial camera placement and setup.

### 3.2 `on frame`

Runs every frame. This is the most common place for animation and per-frame updates.

### 3.3 `on every <seconds>`

Runs repeatedly with a fixed interval. The engine accumulates `dt` and executes the block in batches if the frame time exceeds the interval.

### 3.4 `on key <key>`

Runs when the user presses a key. The parser accepts keys such as:

- `A`-`Z`
- `0`-`9`
- `F1`-`F12`
- `SPACE`, `ENTER`, `TAB`, `LEFT`, `RIGHT`, `UP`, `DOWN`

### 3.5 `on collide <A> <B>`

Runs when a solid object matching pattern `A` begins overlapping a solid object matching pattern `B`.

The parser stores this as a collision event with:

- `a`: first match pattern
- `b`: second match pattern
- `active`: a set of currently-overlapping name pairs

Inside a collision event, the engine provides substitutions:

- `@a` resolves to the first object name
- `@b` resolves to the second object name
- `$hitx`, `$hity`, `$hitz` hold the midpoint between the colliding objects

Example:

```scr
on collide sweeper crate*
  print sweeper hit @b near $hitx $hity $hitz
  set @b color 1 0.2 0.2
  set sweeper pos -2.3 0.15 1.5
end
```

This is a useful pattern for gameplay effects, impacts, and scripted reactions.

## 4. Built-in variables

The runtime fills a number of variables automatically before each event or command runs:

- `$time`
- `$dt`
- `$frame`
- `$camx`
- `$camy`
- `$camz`
- `$camyaw`
- `$campitch`
- `$objects`
- `$lights`

During collision events, the engine also sets:

- `$hitx`
- `$hity`
- `$hitz`

These are useful for animation, camera-driven spawning, and event logic.

## 5. Matching and naming rules

- Object and light names are compared exactly unless the name ends in `*`.
- `name*` matches any object/light whose name begins with `name`.
- The `#` suffix is resolved to an incrementing counter.

Examples:

```scr
destroy crate*
light point lamp# 0 0 0 1 1 1 5
```

This makes it easy to create and destroy groups of generated objects and lights.

## 6. Collision model

The engine approximates every collidable object as an oriented bounding box (OBB), derived from the object's mesh bounds, size, and rotation.

- Only objects with `collide on` are considered in collision tests.
- The camera can also be considered a collision body with default radius `0.3`.
- Collision events fire once when a pair of objects starts overlapping.
- `collider` objects are invisible oriented boxes and participate in player, camera, and scripted object collision.

This keeps the scripting layer simple while still supporting object interaction and room-scale gameplay logic.

## 7. Example: scene script structure

The default scene in [scene.scr](./scene.scr) shows the intended usage pattern.

```scr
# assets
texture checker assets/checker.png
model   torus   assets/torus.obj

# render setup
builtin cornell
builtin pointlights 100
area on

gi cache
denoise 4
collision camera on 0.3

# objects
spawn torus1 torus pos -0.9 3.05 -0.7 size 1.1 1.1 1.1 tex checker
spawn ball sphere pos 0.95 1.9 0.5 size 0.9 0.9 0.9 color 0.9 0.8 0.3

# events
on start
  camera pos 0 2.5 8.5
  camera look 0 0
end

on frame
  calc $step $dt * 40
  rotate torus1 0 $step 0
end

on key F
  spawn crate# cube pos $camx $camy $camz size 0.4 0.4 0.4 tex checker
end
```

This demonstrates the main pattern of the language:

1. declare assets and built-ins
2. create objects and lights
3. set render settings
4. define behavior using `on ...` blocks
5. use `$variables` and `@name` substitutions for dynamic scene logic

## 8. Execution semantics

The runtime executes script commands in order, and event handlers are just stored blocks that are invoked later.

The logic is roughly:

- parse file into a list of top-level lines and event blocks
- run top-level commands once
- on `start`, fire `on start` blocks
- during update, run `on every`, `on frame`, and key handlers
- detect overlapping solid objects and trigger `on collide` blocks

This design makes the language ideal for scene construction, animation, and runtime event-driven scripting without turning into a full app scripting engine.

## 9. Practical notes

- The language is line-oriented and whitespace-tokenized.
- Commands are resolved by the first token only.
- Most commands accept a fixed number of arguments; invalid forms raise a script error at the source line number.
- The engine is permissive about many scene-editing operations and intentionally built around object- and light-centric scene composition.

## 10. Summary

The SCR language is a compact scene-scripting DSL designed for real-time rendering and object behavior. It is built around:

- declarative scene creation
- runtime event-driven blocks
- lightweight variable arithmetic
- collision-driven interaction
- camera and lighting control

If you want to extend it, the primary implementation point is [world.hpp](./world.hpp), especially the `Script::loadText`, `Script::command`, `Script::run`, and `Script::updateCollisions` logic.
