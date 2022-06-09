# Gradient doodle

I wanted to make myself a little proud and created this.
It examplifies back propagation by showing how an arm moves towards
a target. Its position is determined by its angles between its fixed sized
arms and its movement is connected to the learning rate.
Model fitting is done by gradient descent or ADAM optimization (a gradient based scheme)
where the solution is unknown. There is no guaranteed solution,
sometimes it reaches the target and sometimes it doesn't. It does
seem like ADAM works better.

# Modeling an arm

I model an arm using angles and developed a mathematical model for that purpose.
Backpropagation made possible after some pen and paper work. I derived all of this
on a series of small physical post-it notes. 

```C
// model in short
p[n] = (p_x[n], p_y[n])

p_x[n] = p_x[n-1] + r_n * cos( sum(omegas up to n) + (n-1) * pi )
p_y[n] = p_y[n-1] + r_n * sin( sum(omegas up to n) + (n-1) * pi ) 

p[0] = /* start position of the arm */

/* Loss calculated as distance to the power of two time a half */
// t == target

E = 1/2 * (p - t) ^ 2
dE/dPn = (p - t) * ( dP/dOmega_q ) = ( (px[n] - tx) * dp_x[n]/dOmega_q , (py[n] - ty) * dp_y[n]/dOmega_q )

dP_x[n]/dOmega_q = 0 if q > n otherwise dP_x[n-1]/dOmega_q - r_n * sin( sum(omegas up to n) + (n-1) * pi)
dP_y[n]/dOmega_q = 0 if q > n otherwise dP_y[n-1]/dOmega_q + r_n * cos( sum(omegas up to n) + (n-1) * pi) 
```

# Test

Download the repo and open the index.html.

Or visit this link to where I host the project: [demo](https://me.ricardicus.se/7ef9bb61b8d8740133e7cb423e9e28e3)




