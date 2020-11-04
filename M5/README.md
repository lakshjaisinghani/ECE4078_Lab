# Group 1_04 Milestone 5 
### No SDF files have been modified

## Instructions:
Once the environment is intitialised, starting our implementtion requires you to run a single python3 script:
`python autoNavigation.py`

The PenguinPi should then navigate the environment autonomously.

## Controls for teleoperation
Teleoperation can be picked up at any time and reverts back to autonav when you press the spacebar

- Up Arrow: Forward

- Down Arrow: Reverse

- Left Arrow: Turn Left 

- Right Arrow: Turn Right 

- Spacebar: Stop the bot

Once the PPi has seen all aruco markers the script can be stopped via cmd line and a txt file (team_1_04.txt) will print the estimated marker and object locations as well as the available paths from each marker to other markers and the estimated distance between,

## Caution!!
Our M5 implementation has been tested on native Ubuntu at ~50fps (although our model isnt reliant on fps readings). We have expereienced issue with our model when screensharing over zoom, which is of concern especially given the live demo format. We believe this is due to the compute power of the computer used not being able to keep up with slam updates, and our PPi signals continue to be sent even with our slam updates lagging. This results in a slow theta value significantly degrading the quality our implementation since the robot angle is an integral component.
If this happens we would request that would implementation could be run natively on a demonstrators computer with either more compute power or without screen-recording.


### Kind regards,
### Team 1_04 :)