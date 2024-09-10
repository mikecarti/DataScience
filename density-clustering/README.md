## My implementation of OPTICS and DBSCAN algorithms. 

Special thanks to Kirill Kiosa and Roman Kholinov for data preparation stage.

## Usage: 

### DBSCAN
```python
db = DataBase()

db.fit([[1, 2], [0, 0], [3, 3], [5, 8], [9,9]])

dbscan(db, eps_radius=5, min_pts_density=1)
print(db.stats())
>Cluster 0 has 3 points in it
>Cluster 1 has 2 points in it

# plots 2d scatterplot for 2 columns from dataset
db.plot_clusters("xlabel", "ylabel", col1=0, col2=1)
```

### OPTICS
```python
db = OpticsDataBase()
db.fit([[1, 2], [0, 0], [3, 3], [5, 8], [9,9]])
optics(db, eps_radius=3, min_pts_density=5)
db.plot_reachability()
```

## Example
![image](https://github.com/user-attachments/assets/2d658eba-1b04-4067-b6ef-8f697209a5e6)

![image](https://github.com/user-attachments/assets/45b78c5c-d6e9-4788-899d-6e889b2f04c0)


