# include <sodeneuralprogram.h>
# include <math.h>
# define LAMBDA	100.0
typedef double(*DOUBLE_FUNCTION)();
typedef int(*INTEGER_FUNCTION)();


SodeNeuralProgram::SodeNeuralProgram(QString filename)
    :NeuralProgram()
{
    int i;
    ptr=new QLibrary(filename);
    if(ptr==NULL)
    {
        x0=0.0;
        x1=1.0;
        npoints=10;
        node=1;
        setDimension(node);
        enableMultiple(node);
        f0=new double[node];
        systemfun=NULL;
        fsystemfun=NULL;
        systemf0=NULL;
        fsystemf0=NULL;
        parser.resize(node);
        for(int i=0;i<node;i++) parser[i]=new FunctionParser();
    }
    else
    {
        DOUBLE_FUNCTION X0, X1;
        INTEGER_FUNCTION NODE, NPOINTS;

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

        NODE=(INTEGER_FUNCTION)ptr->resolve("getnode");
        if(NODE==NULL) NODE=(INTEGER_FUNCTION)ptr->resolve("_getnode");
        if(NODE==NULL) NODE=(INTEGER_FUNCTION)ptr->resolve("getnode_");
        if(NODE==NULL) NODE=(INTEGER_FUNCTION)ptr->resolve("_getnode_");
        if(NODE==NULL) node=1;else node=NODE();
        setDimension(node);
        enableMultiple(node);
        NPOINTS=(INTEGER_FUNCTION)ptr->resolve("getnpoints");
        if(NPOINTS==NULL)
            NPOINTS=(INTEGER_FUNCTION)ptr->resolve("_getnpoints");
        if(NPOINTS==NULL)
            NPOINTS=(INTEGER_FUNCTION)ptr->resolve("getnpoints_");
        if(NPOINTS==NULL)
            NPOINTS=(INTEGER_FUNCTION)ptr->resolve("_getnpoints_");
        if(NPOINTS==NULL) npoints=10; else npoints=NPOINTS();

        fsystemfun=NULL;
        fsystemf0=NULL;
        systemfun=(GSYSTEMFUN)ptr->resolve("systemfun");
        if(systemfun==NULL)
            systemfun=(GSYSTEMFUN)ptr->resolve("_systemfun");
        if(systemfun==NULL)
            fsystemfun=(GFSYSTEMFUN)ptr->resolve("systemfun_");
        if(fsystemfun==NULL)
            fsystemfun=(GFSYSTEMFUN)ptr->resolve("systemfun_");
        systemder=(GSYSTEMDER)ptr->resolve("systemder");
        if(systemder==NULL)systemder=(GSYSTEMDER)ptr->resolve("_systemder");
        if(systemder==NULL)fsystemder=(GFSYSTEMDER)ptr->resolve("systemder_");
        if(fsystemder==NULL) fsystemder=(GFSYSTEMDER)ptr->resolve("_systemder_");

        systemf0=(GSYSTEMF0)ptr->resolve("systemf0");
        if(systemf0==NULL)
            systemf0=(GSYSTEMF0)ptr->resolve("_systemf0");
        if(systemf0==NULL)
        {
            fsystemf0=(GFSYSTEMF0)ptr->resolve("systemf0_");
            if(fsystemf0==NULL)
                fsystemf0=(GFSYSTEMF0)ptr->resolve("_systemf0_");
            f0=new double[node];
            fsystemf0(&node,f0);
        }
        else
        {
            f0=new double[node];
            systemf0(node,f0);
        }
        parser.resize(node);
        for(i=0;i<node;i++) parser[i]=new FunctionParser();
    }

	currentparser=0;
    trainx.resize(npoints);
    testx.resize(10*npoints);
	for(int i=0;i<trainx.size();i++)
	{
		trainx[i].resize(1);
        double a=getX0();
        double b=getX1();
		double step=(b-a)/100.0;
		if(i==0) trainx[i][0]=a; else trainx[i][0]=trainx[i-1][0]+step;
	}
	for(int i=0;i<testx.size();i++)
	{
		testx[i].resize(1);
        double a=getX0();
        double b=getX1();
		double step=(b-a)/1000.0;
		if(i==0) testx[i][0]=a; else testx[i][0]=testx[i-1][0]+step;
	}
    f0=new double[getNode()];
    systemf0(getNode(),f0);
    y=new double[getNode()];
    yy=new double[getNode()];
    res=new double[getNode()];
}

double	SodeNeuralProgram::penalty1()
{
	double sum=0.0;
    for(int i=0;i<getNode();i++)
	{
		double v=nparser[i]->eval(trainx[0])-f0[i];
		sum=sum+v*v;
	}
	return sum;
}

double	SodeNeuralProgram::penalty2()
{
	return 0.0;
}

double	SodeNeuralProgram::penalty3()
{
	return 0.0;
}

double	SodeNeuralProgram::penalty4()
{
	return 0.0;
}


void	SodeNeuralProgram::getDeriv(Data &g)
{

	for(int i=0;i<g.size();i++) g[i]=0.0;
	Data tempg;
	Data tempg2;
	tempg.resize(g.size());
	tempg2.resize(g.size());
	Data w;
	neuralparser->getWeights(w);
    double *res2=new double[getNode()];
	for(int i=0;i<trainx.size();i++)
	{
        for(int j=0;j<getNode();j++)
		{
			y[j]=nparser[j]->eval(trainx[i]);
			yy[j]=nparser[j]->evalDeriv(trainx[i],1);
		}
        systemfun(getNode(),trainx[i][0],y,yy,res);
		neuralparser->getDeriv(trainx[i],tempg);
		neuralparser->getXDeriv(trainx[i],1,tempg2);
		
		for(int j=0;j<g.size();j++) 
		{
			systemder(currentparser,trainx[i][0],y,yy,tempg[j],tempg2[j],res2);
            for(int k=0;k<getNode();k++) g[j]+=2.0*res[k]*res2[k];
		}
	}
	double v=neuralparser->eval(trainx[0])-f0[currentparser];
	neuralparser->getDeriv(trainx[0],tempg);
	for(int j=0;j<g.size();j++)
	{
		double out=tempg[j];
		g[j]+=2.0*LAMBDA*v*out;
	}
	delete[] res2;


}

double	SodeNeuralProgram::getPartError(Data &value)
{
    for(int i=0;i<getNode();i++) value[i]=0.0;
	for(int i=0;i<trainx.size();i++)
	{
		double v;
        for(int j=0;j<getNode();j++)
		{
			y[j]=nparser[j]->eval(trainx[i]);
			yy[j]=nparser[j]->evalDeriv(trainx[i],1);
		}
        systemfun(getNode(),trainx[i][0],y,yy,res);
        for(int j=0;j<getNode();j++)
		{
			v=res[j];
			value[j]=value[j]+v*v;
		}
	}
    for(int j=0;j<getNode();j++) value[j]=value[j]+LAMBDA*pow(nparser[j]->eval(trainx[0])-f0[j],2.0);
	return 0.0;
}

double	SodeNeuralProgram::getPartError()
{
	double value=0.0;

	for(int i=0;i<trainx.size();i++)
	{
		double v;
        for(int j=0;j<getNode();j++)
		{
			y[j]=nparser[j]->eval(trainx[i]);
			yy[j]=nparser[j]->evalDeriv(trainx[i],1);
		}
        systemfun(getNode(),trainx[i][0],y,yy,res);
		v=res[currentparser];
		value=value+v*v;
	}

	return value+LAMBDA*pow(neuralparser->eval(trainx[0])-f0[currentparser],2.0);
}

double	SodeNeuralProgram::getTrainError()
{

	double value=0.0;

	for(int i=0;i<trainx.size();i++)
	{
		double v;
        for(int j=0;j<getNode();j++)
		{
			y[j]=nparser[j]->eval(trainx[i]);
			yy[j]=nparser[j]->evalDeriv(trainx[i],1);
		}
        systemfun(getNode(),trainx[i][0],y,yy,res);
		v=0.0;
        for(int j=0;j<getNode();j++) v+=res[j]*res[j];
		value=value+v;
	}

	double penalty=0.0;
	penalty=penalty1();
	return value+LAMBDA*penalty;
}

double	SodeNeuralProgram::getX0() const
{
    /*	Return the left boundary of the equations.
     * */
    return x0;
}

double	SodeNeuralProgram::getX1() const
{
    /*	Return the right boundary of the equations.
     * */
    return x1;
}

int	SodeNeuralProgram::getNode() const
{
    /*	Return the amount of ODE's in the system.
     * */
    return node;
}

int	SodeNeuralProgram::getNpoints() const
{
    /*	Return the amount of training points.
     * */
    return npoints;
}


double	SodeNeuralProgram::getTestError()
{
    double value=0.0,v=0.0;
	for(int i=0;i<testx.size();i++)
	{
        for(int j=0;j<getNode();j++)
		{
			y[j]=nparser[j]->eval(testx[i]);
			yy[j]=nparser[j]->evalDeriv(testx[i],1);
		}
        systemfun(getNode(),testx[i][0],y,yy,res);
		v=0.0;
        for(int j=0;j<getNode();j++) v=v+res[j]*res[j];
		value=value+v;
	}
	return value;
}

void    SodeNeuralProgram::printOutput(QString filename)
{
    QFile fp(filename);
    if(!fp.open(QIODevice::WriteOnly |QIODevice::Text)) return;
    QTextStream st(&fp);
    for(int i=0;i<testx.size();i++)
    {
        st<<testx[i][0]<<" ";
        for(int j=0;j<getNode();j++)
        {
            y[j]=nparser[j]->eval(testx[i]);
            yy[j]=nparser[j]->evalDeriv(testx[i],1);
            st<<y[j]<<" ";
        }
        systemfun(getNode(),testx[i][0],y,yy,res);
        for(int j=0;j<getNode();j++)
        {
            st<<res[j]<<" ";
        }
      st<<endl;
    }
    fp.close();
}

SodeNeuralProgram::~SodeNeuralProgram()
{
	delete[] f0;
	delete[] y;
	delete[] yy;
	delete[] res;
}
