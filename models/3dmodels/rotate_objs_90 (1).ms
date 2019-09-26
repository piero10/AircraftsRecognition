-- rotate selected objects by +-90 degrees
if rol_rot90 != undefined do
	if classOf rol_rot90 == RolloutClass do
		destroyDialog rol_rot90
rollout rol_rot90 "Rotate Me"
(
	local axis = [1,0,0]
	
	group "Rotate Axis"
	(
		radioButtons ax labels:#("X","Y","Z") columns:3
	)
	group "Rotate Direction"
	(
		button rotNeg "-90" width:60 across:2
		button rotPos "+90" width:60
	)
	on ax changed state do
	(
		case state of (
			1: axis = [1,0,0]
			2: axis = [0,1,0]
			3: axis = [0,0,1]
		)
	)
	on rotNeg pressed do (rotate selection -90 axis)
	on rotPos pressed do (rotate selection 90 axis)
)
createDialog rol_rot90 162 101 \
style:#(#style_titlebar, #style_border, #style_sysmenu, #style_minimizebox)
