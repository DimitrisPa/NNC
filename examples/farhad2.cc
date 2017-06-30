# include <math.h>
/*	This is a sample file for ODE,
 *	written in C++. The meaning of the functions is 
 *	as follows:
 *		1. getx0():  Return the left boundary of the equation.
 *		2. getx1():  Return the right boundary of the equation.
 *		3. getkind():Return the kind of the equation.
 *		4. getnpoints(): Return the number points in which the system
 *				will try to solve the ODE.
 *		5. getf0():  Return the left boundary condition.
 *		6. getf1():  Return the right boundary condition.
 *		7. getff0(): Return the derivative of the left boundary condition.
 *		8. ode1ff(): Return the equation for first order equations.
 *		9. ode2ff(): Return the equation for second order equations.
 * */
extern "C"
{

double	getx0()
{
	return 0;
}

double	getx1()
{
	return 1;
}

int	getkind()
{
	return 1;
}

int	getnpoints()
{
	return 20;
}

double	getf0()
{
	return 1.0;
}

double	getff0()
{
	return 3.0;
}

double	getf1()
{
	return sin(10.0);
}

double	ode1ff(double x,double y,double yy)
{
	double ff=(1+3*x*x)/(1+x+x*x*x);
	return yy+(x+ff)*y-x*x*x-2*x-x*x*ff;
}

double	dode1ff(double x,double y,double yy,double dy,double dyy)
{
	double ff=(1+3*x*x)/(1+x+x*x*x);
	double dff=6*x*(1+x+x*x*x)-(1+3*x*x)*(1+3*x*x)/pow(1+x+x*x*x,2.0);
	return dyy+(x+ff)*dy+(1+dff)*y-3*x*x-2.0-2*x*ff-x*x*dff;
}

double	ode2ff(double x,double y,double yy,double yyy)
{
	return yyy+cos(x)*y;
}

double	dode2ff(double x,double y,double yy,double yyy,double dy,double dyy,double dyyy)
{
	return dyyy+sin(x)*y+dy*cos(x);
}

}
