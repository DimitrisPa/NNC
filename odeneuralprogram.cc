# include <odeneuralprogram.h>
# include <math.h>
# include <QFile>
# include <QTextStream>
# include <QIODevice>
typedef vector<double> Data;
# define	LAMBDA	1000.0


OdeNeuralProgram::OdeNeuralProgram(QString filename)
	:NeuralProgram(1)
{

    ptr=new QLibrary(filename);
    DOUBLE_FUNCTION X0,X1,F0,F1,FF0;
    INTEGER_FUNCTION NPOINTS, KIND;

    KIND=(INTEGER_FUNCTION)ptr->resolve("getkind");
    if(KIND==NULL) KIND=(INTEGER_FUNCTION)ptr->resolve("_getkind");
    if(KIND==NULL) KIND=(INTEGER_FUNCTION)ptr->resolve("getkind_");
    if(KIND==NULL) KIND=(INTEGER_FUNCTION)ptr->resolve("_getkind_");
    if(KIND==NULL) kind=ODE1; else kind=KIND();

    NPOINTS=(INTEGER_FUNCTION)ptr->resolve("getnpoints");
    if(NPOINTS==NULL) NPOINTS=(INTEGER_FUNCTION)ptr->resolve("_getnpoints");
    if(NPOINTS==NULL) NPOINTS=(INTEGER_FUNCTION)ptr->resolve("getnpoints_");
    if(NPOINTS==NULL) NPOINTS=(INTEGER_FUNCTION)ptr->resolve("_getnpoints_");
    if(NPOINTS==NULL) npoints=10; else npoints=NPOINTS();
    trainx.resize(npoints);
    testx.resize(10 * npoints);

    X0=(DOUBLE_FUNCTION)ptr->resolve("getx0");
    if(X0==NULL) X0=(DOUBLE_FUNCTION)ptr->resolve("_getx0");
    if(X0==NULL) X0=(DOUBLE_FUNCTION)ptr->resolve("getx0_");
    if(X0==NULL) X0=(DOUBLE_FUNCTION)ptr->resolve("_getx0_");
    if(X0==NULL) x0=0.0; else x0=X0();

    X1=(DOUBLE_FUNCTION)ptr->resolve("getx1");
    if(X1==NULL) X1=(DOUBLE_FUNCTION)ptr->resolve("_getx1");
    if(X1==NULL) X1=(DOUBLE_FUNCTION)ptr->resolve("getx1_");
    if(X1==NULL) X1=(DOUBLE_FUNCTION)ptr->resolve("_getx1_");
    if(X1==NULL) x1=1.0; else x1=X1();

    F0=(DOUBLE_FUNCTION)ptr->resolve("getf0");
    if(F0==NULL) F0=(DOUBLE_FUNCTION)ptr->resolve("_getf0");
    if(F0==NULL) F0=(DOUBLE_FUNCTION)ptr->resolve("getf0_");
    if(F0==NULL) F0=(DOUBLE_FUNCTION)ptr->resolve("_getf0_");
    if(F0==NULL) f0=0.0; else f0=F0();

    F1=(DOUBLE_FUNCTION)ptr->resolve("getf1");
    if(F1==NULL) F1=(DOUBLE_FUNCTION)ptr->resolve("_getf1");
    if(F1==NULL) F1=(DOUBLE_FUNCTION)ptr->resolve("getf1_");
    if(F1==NULL) F1=(DOUBLE_FUNCTION)ptr->resolve("_getf1");
    if(F1==NULL) f1=0.0; else f1=F1();

    FF0=(DOUBLE_FUNCTION)ptr->resolve("getff0");
    if(FF0==NULL) FF0=(DOUBLE_FUNCTION)ptr->resolve("_getff0");
    if(FF0==NULL) FF0=(DOUBLE_FUNCTION)ptr->resolve("getff0_");
    if(FF0==NULL) FF0=(DOUBLE_FUNCTION)ptr->resolve("_getff0_");
    if(FF0==NULL) ff0=0.0; else ff0=FF0();

    fode1ff=NULL;
    fode2ff=NULL;

    ode1ff=(GODE1FF)ptr->resolve("ode1ff");
    if(ode1ff==NULL) ode1ff=(GODE1FF)ptr->resolve("_ode1ff");
    if(ode1ff==NULL) fode1ff=(GFODE1FF)ptr->resolve("ode1ff_");
    if(fode1ff==NULL)fode1ff=(GFODE1FF)ptr->resolve("_ode1ff_");

    dode1ff =(GDODE1FF)ptr->resolve("dode1ff");
    if(dode1ff==NULL) dode1ff=(GDODE1FF)ptr->resolve("_dode1ff");
    if(dode1ff==NULL) fdode1ff=(GFDODE1FF)ptr->resolve("dode1ff_");
    if(fdode1ff==NULL) fdode1ff=(GFDODE1FF)ptr->resolve("_dode1ff_");

    ode2ff=(GODE2FF)ptr->resolve("ode2ff");
    if(ode2ff==NULL) ode2ff=(GODE2FF)ptr->resolve("_ode2ff");
    if(ode2ff==NULL) fode2ff=(GFODE2FF)ptr->resolve("ode2ff_");
    if(fode2ff==NULL) fode2ff=(GFODE2FF)ptr->resolve("_ode2ff_");

	for(int i=0;i<trainx.size();i++)
	{
		trainx[i].resize(1);
        double a=getX0();
        double b=getX1();
		double step=(b-a)/100;
		if(i==0) trainx[i][0]=a; else trainx[i][0]=trainx[i-1][0]+step;
//		valx[i].resize(1);
 //       valx[i][0]=getX0()+(getX1()-getX0())*drand48();
	}
	for(int i=0;i<testx.size();i++)
	{
		testx[i].resize(1);
        double a=getX0();
        double b=getX1();
		double step=(b-a)/1000.0;
		if(i==0) testx[i][0]=a; else testx[i][0]=testx[i-1][0]+step;
	}


}

double	OdeNeuralProgram::penalty1()
{
    return LAMBDA * pow(neuralparser->eval(trainx[0])-getF0(),2.0);
}

double	OdeNeuralProgram::penalty2()
{

    return	LAMBDA * pow(neuralparser->evalDeriv(trainx[0],1)-getFF0(),2.0);
}

double	OdeNeuralProgram::penalty3()
{
	Data point;
	point.resize(1);
    point[0]=getX1();
    return LAMBDA * pow(neuralparser->eval(point)-getF1(),2.0);
}

double	OdeNeuralProgram::penalty4()
{
	return 0.0;
}

void	OdeNeuralProgram::getDeriv(Data &g)
{
	for(int i=0;i<g.size();i++) g[i]=0.0;
	Data tempg,tempg2,tempg3;
	tempg.resize(g.size());
	tempg2.resize(g.size());
	tempg3.resize(g.size());
	double v,v1,v2,v3;
	for(int i=0;i<trainx.size();i++)
	{
        if(getKind()==1)
		{
			v1=neuralparser->eval(trainx[i]);
			v2=neuralparser->evalDeriv(trainx[i],1);
			v=ode1ff(trainx[i][0],v1,v2);
			neuralparser->getDeriv(trainx[i],tempg);
			neuralparser->getXDeriv(trainx[i],1,tempg2);
			double outg=0;
			for(int j=0;j<g.size();j++)
			{
				outg=dode1ff(trainx[i][0],v1,v2,tempg[j],tempg2[j]);
				g[j]+=2.0*v*outg;
			}
		}
		else
        if(getKind()==2)
		{
			v1=neuralparser->eval(trainx[i]);
			v2=neuralparser->evalDeriv(trainx[i],1);
			v3=neuralparser->evalDeriv2(trainx[i],1);
			v=ode2ff(trainx[i][0],v1,v2,v3);
			neuralparser->getDeriv(trainx[i],tempg);
			neuralparser->getXDeriv(trainx[i],1,tempg2);
			neuralparser->getX2Deriv(trainx[i],1,tempg3);

			double outg=0;
			for(int j=0;j<g.size();j++)
			{
				outg=dode2ff(trainx[i][0],v1,v2,v3,tempg[j],tempg2[j],tempg3[j]);
				g[j]+=2.0*v*outg;
			}
		}
		else
        if(getKind()==3)
		{
			v1=neuralparser->eval(trainx[i]);
			v2=neuralparser->evalDeriv(trainx[i],1);
			v3=neuralparser->evalDeriv2(trainx[i],1);
			v=ode2ff(trainx[i][0],v1,v2,v3);
			neuralparser->getDeriv(trainx[i],tempg);
			neuralparser->getXDeriv(trainx[i],1,tempg2);
			neuralparser->getX2Deriv(trainx[i],1,tempg3);
			double outg=0;
			for(int j=0;j<g.size();j++)
			{
				outg=dode2ff(trainx[i][0],v1,v2,v3,tempg[j],tempg2[j],tempg3[j]);
				g[j]+=2.0*v*outg;
			}
		}
	}
    if(getKind()==1)
	{
		v1=neuralparser->eval(trainx[0]);
		neuralparser->getDeriv(trainx[0],tempg);
		for(int j=0;j<g.size();j++)
		{
			double outg=0;
			outg=tempg[j];
            g[j]+=2.0*LAMBDA*(v1-getF0())*outg;
		}
	}
	else
    if(getKind()==2)
	{
		v1=neuralparser->eval(trainx[0]);
		v2=neuralparser->evalDeriv(trainx[0],1);
		neuralparser->getDeriv(trainx[0],tempg);
		neuralparser->getXDeriv(trainx[0],1,tempg2);
		for(int j=0;j<g.size();j++)
		{

            g[j]+=2.0*LAMBDA*(v1-getF0())*tempg[j]+2.0*LAMBDA*(v2-getFF0())*tempg2[j];
		}
	}
	else
    if(getKind()==3)
	{
		v1=neuralparser->eval(trainx[0]);
		neuralparser->getDeriv(trainx[0],tempg);
        Data point;point.resize(1);point[0]=getX1();
		v2=neuralparser->eval(point);
		neuralparser->getDeriv(point,tempg2);
		for(int j=0;j<g.size();j++)
		{
			double outg=0;
            g[j]+=2.0*LAMBDA*(v1-getF0())*tempg[j]+2.0*LAMBDA*(v2-getF1())*tempg2[j];
		}
	}
}


double	OdeNeuralProgram::getTrainError()
{
	double value=0.0;
	double maxv=-1;

	for(int i=0;i<trainx.size();i++)
	{
		double v;
		double v1,v2,v3;
        if(getKind()==1)
		{
			v1=neuralparser->eval(trainx[i]);
			v2=neuralparser->evalDeriv(trainx[i],1);
			v=ode1ff(trainx[i][0],v1,v2);
		}
		else
        if(getKind()==2)
		{
			v1=neuralparser->eval(trainx[i]);
			v2=neuralparser->evalDeriv(trainx[i],1);
			v3=neuralparser->evalDeriv2(trainx[i],1);
			v=ode2ff(trainx[i][0],v1,v2,v3);
		}
		else
        if(getKind()==3)
		{
			v1=neuralparser->eval(trainx[i]);
			v2=neuralparser->evalDeriv(trainx[i],1);
			v3=neuralparser->evalDeriv2(trainx[i],1);
			v=ode2ff(trainx[i][0],v1,v2,v3);
		}
		if(fabs(v) > maxv) maxv=fabs(v);
		value=value+v*v;
	}
	double penalty=0.0;
    if(getKind()==1)	penalty=penalty1();
	else
    if(getKind()==2)	penalty=penalty1()+penalty2();
	else
    if(getKind()==3)	penalty=penalty1()+penalty3();
	return value+penalty;
}

double	OdeNeuralProgram::getTestError()
{
	double value=0.0;
	for(int i=0;i<testx.size();i++)
	{
		double v,v1,v2,v3;
        if(getKind()==1)
		{
			v1=neuralparser->eval(testx[i]);
			v2=neuralparser->evalDeriv(testx[i],1);
			v=ode1ff(testx[i][0],v1,v2);
		}
		else
        if(getKind()==2)
		{
			v1=neuralparser->eval(testx[i]);
			v2=neuralparser->evalDeriv(testx[i],1);
			v3=neuralparser->evalDeriv2(testx[i],1);
			v=ode2ff(testx[i][0],v1,v2,v3);
		}
		else
        if(getKind()==3)
		{
			v1=neuralparser->eval(testx[i]);
			v2=neuralparser->evalDeriv(testx[i],1);
			v3=neuralparser->evalDeriv2(testx[i],1);
			v=ode2ff(testx[i][0],v1,v2,v3);
		}
		value=value+v*v;
	}
	return value;
}

int	OdeNeuralProgram::getKind()	const
{
    /*	Return the kind of the equation.
     * */
    return kind;
}

double	OdeNeuralProgram::getX0()		const
{
    /*	Return the left boundary.
     * */
    return	x0;
}

double	OdeNeuralProgram::getX1()		const
{
    /*	Return the right boundary.
     * */
    return x1;
}

double	OdeNeuralProgram::getF0()		const
{
    /*	Return the left boundary condition.
     * */
    return	f0;
}

double	OdeNeuralProgram::getF1()		const
{
    /*	Return the right boundary condition.
     * */
    return	f1;
}

double	OdeNeuralProgram::getFF0()	const
{
    /*	Return the derivative of the left
     *	boundary condition.
     * */
    return	ff0;
}

void    OdeNeuralProgram::printOutput(QString filename)
{
    QFile fp(filename);
    if(!fp.open(QIODevice::WriteOnly |QIODevice::Text)) return;
    QTextStream st(&fp);
    double v;
    for(int i=0;i<testx.size();i++)
    {
        double v1,v2,v3;
        if(getKind()==1)
        {
            v1=neuralparser->eval(testx[i]);
            v2=neuralparser->evalDeriv(testx[i],1);
            v=ode1ff(testx[i][0],v1,v2);
        }
        else
        if(getKind()==2)
        {
            v1=neuralparser->eval(testx[i]);
            v2=neuralparser->evalDeriv(testx[i],1);
            v3=neuralparser->evalDeriv2(testx[i],1);
            v=ode2ff(testx[i][0],v1,v2,v3);
        }
        else
        if(getKind()==3)
        {
            v1=neuralparser->eval(testx[i]);
            v2=neuralparser->evalDeriv(testx[i],1);
            v3=neuralparser->evalDeriv2(testx[i],1);
            v=ode2ff(testx[i][0],v1,v2,v3);
        }
        st<<testx[i][0]<<" "<<v1<<" "<<v2<<endl;
    }
    fp.close();
}

OdeNeuralProgram::~OdeNeuralProgram()
{
    if(ptr!=NULL) delete ptr;
}
