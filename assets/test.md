# scene.scr - default scene. Edit and press F5 in the running app to reload.

# ---- assets (paths are relative to this file)
texture checker assets/checker.png
model   torus   assets/torus.obj

# ---- the Cornell room + 512 small point lights + 8 spot lights
builtin cornell
builtin pointlights 1
builtin spots
# builtin sun

# ---- global illumination + light options
gi cache          # path | cache | screen   (N cycles modes while running, H = GI rate, L = area lights)
area on           # off: emissive panels stop emitting, only point / spot / sun lights remain

# ---- NRD denoising + collisions
denoise 4         # 0 = off, any positive value enables NRD (B toggles it while running)
collision camera on 0.3   # the camera can no longer fly through walls / objects (K toggles it)

# ---- objects: spawn <name> <cube|quad|sphere|modelId> pos x y z ...
spawn torus1  torus  pos -0.9 3.05 -0.7 size 1.1 1.1 1.1 tex checker
spawn ball    sphere pos  0.95 1.9 0.5 size 0.9 0.9 0.9 color 0.9 0.8 0.3
spawn crate0  cube   pos -1.7 0.25 1.5 size 0.5 0.5 0.5 tex checker uvscale 2
spawn sweeper cube   pos -2.3 0.15 1.5 size 0.3 0.3 0.3 color 0.2 0.5 0.9

# ---- events ---------------------------------------------------------------
on start
  camera pos 0 2.5 8.5
  camera look 0 0
  print scene loaded - F: crate, X: remove crates, V: random light, R: clear lights, Z/U: area lights off/on
end

# camera lock on

# every frame: spin the torus (moving objects rebuilds the BLAS each frame - fine for small scenes)
on frame
  calc $step $dt * 40
  rotate torus1 0 $step 0
end

# F drops a textured crate at the camera
on key F
  spawn crate# cube pos $camx $camy $camz size 0.4 0.4 0.4 tex checker
end

on key X
  destroy crate*
end

# V adds a random coloured point light
on key V
  rand $x -2 2
  rand $y 0.5 4.5
  rand $z -2 2
  rand $r 0.2 1
  rand $g 0.2 1
  rand $b 0.2 1
  light point lamp# $x $y $z $r $g $b 0.1 6 0.05
end

on key R
  destroy lamp*
end

# Z / U: area lights off / on (compare point-light-only GI)
on key Z
  area off
end

on key U
  area on
end

# collision event demo: a small cube slides along the floor into crate0, prints where they met, turns the
# crate red and starts over. (@a = first object, @b = second object, $hitx/$hity/$hitz = contact area)
on frame
  calc $slide $dt * 0.8
  move sweeper $slide 0 0
end

on collide sweeper crate*
  print sweeper hit @b near $hitx $hity $hitz
  set @b color 1 0.2 0.2
  set sweeper pos -2.3 0.15 1.5
end

# every 3 seconds: bounce the ball between two heights
on every 3
  if $objects > 0
    set ball pos 0.95 1.9 0.5
  endif
end
