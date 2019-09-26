/*
	* Title: Random Rotate	
	* Author: Preeti Panchal	

	* Description: 
	-- Rotate multiple objects in random directions as per the values chosen
	-- If needed, we can also reset the rotation of objects to [0,0,0]
	-- Undo command is also supported. If the model(s) have custom rotation values then the only way to get this rotation values is to use Undo command
	
	* How to use this script:
	-- Select the model(s)
	-- Set the rotation values
	-- Click on Randomly Rotate button to rotate selected objects
	-- You can also reset the rotation to [0,0,0] values by pressing the Reset rotation values button

	* Created: 14 November 2015	
	* Copyright: (c) Preeti Panchal 2016
	* Licence: Preeti Panchal 
*/
macroScript RandomRotateRollout
	category:"Preeti's Tools" 
	toolTip:"Random Rotate"

(
	try(destroyDialog RandomRotateRollout)catch()
	rollout RandomRotateRollout "Random Rotate" width:248 height:176
	(
		label lbl1 "X-Axis " pos:[18,32] width:48 height:16 
	 
		spinner spn1 "Min:" pos:[76,32] width:67 height:16 range:[-360,360,-3] type:#float
		spinner spn2 "Max:" pos:[164,32] width:67 height:16 range:[-360,360,3] type:#float
		
		label lbl2 "Y-Axis " pos:[18,50] width:48 height:16
		spinner spn3 "Min:" pos:[76,50] width:67 height:16 range:[-360,360,-3] type:#float
		spinner spn4 "Max:" pos:[164,50] width:67 height:16 range:[-360,360,3] type:#float
		
		label lbl3 "Z-Axis " pos:[18,67] width:48 height:16
		spinner spn5 "Min:" pos:[76,67] width:67 height:16 range:[-360,360,0] type:#float
		spinner spn6 "Max:" pos:[164,67] width:67 height:16 range:[-360,360,0] type:#float
		
		GroupBox grp1 "Select rotation values" pos:[8,8] width:232 height:88
		button btn1 " - - - Randomly rotate! - - - " pos:[8,101] width:232 height:32
		button btn2 "Reset rotation to [0,0,0]" pos:[8,137] width:232 height:32

		fn SetObjectRotation obj rx ry rz =
		(
			-- Reset the object's transformation matrix so that 
			-- it only includes position and scale information. 
			-- Doing this clears out any previous object rotation.
			local translateMat = transMatrix obj.transform.pos
			local scaleMat = scaleMatrix obj.transform.scale
			obj.transform = scaleMat * translateMat
		   
			-- Perform each axis rotation individually
			rotate obj (angleaxis rx [1,0,0])
			rotate obj (angleaxis ry [0,1,0])
			rotate obj (angleaxis rz [0,0,1])
		) --Function defined to reset the rotation of selected objects to (0,0,0).
			
		
		on btn1 pressed do
		(
			undo on
			if (selection.count != 0) then
				for o1 in selection do
				(
					randXrot = random spn1.value spn2.value 
					randYrot = random spn3.value spn4.value 
					randZrot = random spn5.value spn6.value 
					o1Rot = eulerAngles randXrot randYrot randZrot
					rotate o1 o1Rot
				)			
			else
				(
					messageBox "Atleast one object should be selected!"
				)
		)
		on btn2 pressed do 
		(	
			undo on
			if (selection.count != 0) then		
				for o2 in selection do
					(
						-- Resets the rotation of selected objects to (0,0,0) by using the previously defined function.
						(SetObjectRotation o2 0 0 0)
					)
			else
				(
					messageBox "Atleast one object should be selected!"
				)
		)
	)
	createDialog RandomRotateRollout
)
