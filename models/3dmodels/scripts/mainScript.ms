folder = "c:\\tmp\\"
num = 2
n = 1/num

for i = 1 to num do 
(
	for j = 1 to num do 
    (
		for k = 1 to num do 
		(
		s1 = i as string
		s2 = j as string
		s3 = k as string			
			
		file = folder+s1+s2+s3+".jpg"
		select objects
		rotate $ (angleaxis 10 [i*n, j*n, k*n])
		render outputfile:file
		)
	)
)