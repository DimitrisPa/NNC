# ifndef __SODENEURALPROGRAM__H
# define __SODENEURALPROGRAM__H
# include <neuralprogram.h>
# include <QLibrary>
typedef vector<double> Data;
/*	GSYSTEMFUN:	Function which represent the system of ODE's. The first
 *			argument is the number of the equations, the second is
 *			the point at which we want to evaluate the system, the third
 *			argument is a table that holds the value of y's and the last
 *			parameter is a table that holds the value of derivatives of y's.
 *	GFSYSTEMFUN:	The same as above, but for the Fortran programming language.
 *	GSYSTEMF0:	Represents the boundary conditions for every ode in the system. The
 *			first parameter is the number of the ODE's and the second is a table
 *			that holds the values of y's at x=x0.
 *	GFSYSTEMF0:	The same as above, but for the Fortran programming language.
 * */
typedef double(*GSYSTEMFUN)(int ,double, double *,double *,double *);
typedef double(*GFSYSTEMFUN)(int *,double *,double *,double *,double *);
typedef void(*GSYSTEMF0)(int,double *);
typedef int(*GFSYSTEMF0)(int *,double *);
typedef void(*GSYSTEMDER)(int,double,double*,double *,double,double,double*);
typedef void(*GFSYSTEMDER)(int*,double*,double*,double *,double*,double*,double*);

class SodeNeuralProgram :
	public NeuralProgram
{
	private:
		vector<Data> trainx;
		vector<Data> testx;
		double	*f0;
		double *y;
		double *yy;
		double *res;
        int 	node;
        double 	x0,x1;
        int 	npoints;
        GSYSTEMFUN 	systemfun;
        GFSYSTEMFUN 	fsystemfun;
        GSYSTEMF0   	systemf0;
        GFSYSTEMF0  	fsystemf0;
        GSYSTEMDER      systemder;
        GFSYSTEMDER     fsystemder;
        QLibrary        *ptr;
        vector<FunctionParser*> parser;
	public:
        SodeNeuralProgram(QString filename);
		int	currentparser;
		virtual double	getPartError();
		virtual double	getPartError(Data &value);
		virtual void	getDeriv(Data &g);
		virtual double	getTrainError();
		virtual double	getTestError();
		virtual double 	penalty1();
		virtual double 	penalty2();
		virtual double 	penalty3();
		virtual double 	penalty4();
        double	getX0() const;
        double	getX1() const;
        int	getNode() const;
        int	getNpoints() const;
        void	getF0(double *f) const;
        void	setGsystemFun(GSYSTEMFUN f);
        virtual void    printOutput(QString filename);
		~SodeNeuralProgram();
};

# endif
