
___________________________________________
**Create subtree for Heroku :**

`git subtree add --prefix mern https://github.com/assane-sakho/projet-si-et-donnees master --squash` 

`git subtree add --prefix ml https://github.com/assane-sakho/projet-si-et-donnees ml --squash`  

---
**Push changes to Heroku :**

`git subtree push --prefix mern https://git.heroku.com/projet-si-et-donnees.git master`  
`git subtree push --prefix ml https://git.heroku.com/projet-si-et-donnees-ml.git master`  

___________________________________________

Run the app using :

`$ docker-compose up --build --remove-orphans`

`$ docker-compose up`

___________________________________________
Mern app cloned from https://github.com/sujaykundu777/mern-docker