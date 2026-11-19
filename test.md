#main scene thing

# material <id> [color r g b] [texture id] [normal id] [smoothness 0..1] [metallic 0..1] [reflectivity 0..1] [ssr on|off]
# Declare normal maps with "texture <id> <file>" and reference them with "normal <id>".
# Assign a material to an object with "spawn ... material <id>" or "set <object> material <id>".

builtin pointlights 10

gi screen
gi offscreen 4
area on

denoise 4

dlss perfomance

texture wall E:\cppprojects\rtengine\assets\wall.png smoothness 0.1
material wall_finish texture wall smoothness 0.1
material orange_metal metallic 0.65 smoothness 0.72 reflectivity 0.35

model Untitled E:\cppprojects\rtengine\assets\Untitled.obj

collider cube2 pos -0.126 -0.150 -0.751 size 2.350 1.000 1.000 rot 0.000 -4.000 0.000
spawn Untitled1 Untitled pos 1.468 0.500 1.298 size 6.000 6.000 6.000 color 0.800 0.800 0.800 material wall_finish
collider cube1 pos 1.734 -1.350 -0.566 size 6.650 1.250 2.250
collider cube3 pos 3.152 -1.400 2.205 size 2.750 1.050 4.400
collider cube4 pos 1.263 -1.450 1.644 size 0.800 1.050 5.550
collider physics_floor pos 0.000 -3.000 0.000 size 40.000 1.000 40.000
spawn physics_box cube pos -3.000 4.000 0.000 size 1.000 1.000 1.000 color 0.900 0.350 0.150 material orange_metal body dynamic mass 1.000 friction 0.500 restitution 0.100

light point lamp1 -2.000 3.000 2.000 1.000 0.950 0.850 12.000


on key V
  rand $x -2 2
  rand $y 0.5 4.5
  rand $z -2 2
  rand $r 0.2 1
  rand $g 0.2 1
  rand $b 0.2 1
  light point lamp# $x $y $z $r $g $b 0.1 6 0.05
end

var $lampX -2.0
var $lampDirection 1

on key J
  calc $lampX $lampX + 2
end

on frame
  calc $lampStep $dt * 0.6
  calc $lampStep $lampStep * $lampDirection
  calc $lampX $lampX + $lampStep
  if $lampX >= -1.5
    var $lampDirection -1
  endif
  if $lampX <= -2.5
    var $lampDirection 1
  endif
  set lamp1 pos $lampX 3 2
end

builtin sun

