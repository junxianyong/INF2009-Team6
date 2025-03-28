# Web Portal User Guide

This user guides contains instructions

## User Roles
A user can have the role of `admin`, `security` or `personnel`.
Only users with the role of `admin` and `security` will be able to access the web portal.
The list of pages available for each user role is as follows.

Admin:
- Dashboard (/)
- Manage Users (/users)
- Manage Mantrap (/mantraps)
- System Logs (/logs)

Security:
- Dashboard (/)
- Manage Mantrap (/mantraps) - Limited features
- System Logs (/logs)

## Login
To login to the GateGuard Security Web Portal, visit https://localhost/. \
URL may change depending on deployment configuration.

![Login Page](images/login.png)

Enter your username and password, then select Login.\
If you forgot your username or password, please contact an administrator to retrieve your username or reset your password.

## Real-Time Monitoring
To monitor the status of the mantrap, select the Dashboard tab on the navigation bar.\
The dashboard will provide a visual representation of the current mantrap status.

System is in idle.
![Dashboard Idle](images/dashboard.png)

Movement has been detected. Waiting for user to approach camera for facial authentication.

![Dashboard Waiting For Face](images/dashboard_waiting_for_face.png)

Detecting and verifying user's face.

![Dashboard Verifying Face](images/dashboard_verifying_face.png)

User has been verified. Door has opened and waiting for user to enter the mantrap.

![Dashboard Waiting For Passage G1](images/dashboard_waiting_for_passage_g1.png)

Detecting the number of people in the mantrap to prevent tailgating.

![Dashboard Checking Mantrap](images/dashboard_checking_mantrap.png)

Re-verifying user's face inside the mantrap.



Verifying user's voice as a second authentication factor.

User has been verified. Door has opened and waiting for the user to exit the mantrap.

## User Management

## Biometrics Enrollment

## Mantrap Management


## Access Logs Monitoring