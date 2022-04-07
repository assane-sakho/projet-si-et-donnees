How to deploy this project on Heroku

1) Crete two projects :
- xxx-mern
- xxx-ml

2) Create a  subtree for Heroku :

`$ git subtree add --prefix mern https://github.com/assane-sakho/projet-si-et-donnees master --squash` 
`$ git subtree add --prefix ml https://github.com/assane-sakho/projet-si-et-donnees ml --squash`  

3) Push changes to Heroku :

`$ git subtree push --prefix mern https://git.heroku.com/xxx-mern.git master`  
`$ git subtree push --prefix ml https://git.heroku.com/xxx-ml.git master`  
___________________________________________
Run the project with Docker compose :

`$ docker-compose build`

You can visist :
- client : http://localhost:3000
- server : http://localhost:8080
- flask(ml) : http://localhost:5000

___________________________________________

Mern app cloned from https://github.com/sujaykundu777/mern-docker
