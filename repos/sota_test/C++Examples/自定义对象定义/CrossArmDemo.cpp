#include "stdafx.h"
#include "CrossArmDemo.h"

#define Property_MainSection                        "MainSectionCurve"
#define Property_AppendSection                      "AppendSectionCurve"
#define Property_BaseWidth                          "BaseWidth"
#define Property_Direction                          "Direction"
#define Property_Height                             "Height"
#define Property_NodeCount                          "NodeCount"
#define Property_EdgeThickness                      "EdgeThickness"
#define Property_BaseThickness                      "BaseThickness"
#define Property_TopThickness                       "TopThickness"

using namespace DemoObject;
using namespace p3d;

CrossArmDemo::CrossArmDemo()
	: m_nWidth(4000)
	, m_bDirection(true)
	, m_nHeight(1000)
	, m_nNode(5)
	, m_nBaseThickness(0)
	, m_nTopThickness(0)
	, m_nEdgeThickness(800)
{
	m_ptrMainSectionCurve = initSectionBase(60);
	m_ptrAppendSectionCurve = initSectionBase(30);
}

CrossArmDemo::~CrossArmDemo()
{

}

p3d::GeCurveArrayPtr DemoObject::CrossArmDemo::getMainSection() const
{
	if (m_ptrMainSectionCurve.isValid())
		return m_ptrMainSectionCurve->clone();
	else
		return nullptr;
}

void DemoObject::CrossArmDemo::setMainSection(p3d::GeCurveArrayPtr ptrSectionCurve)
{
	if (ptrSectionCurve.isValid())
		m_ptrMainSectionCurve = ptrSectionCurve->clone();
}

p3d::GeCurveArrayPtr DemoObject::CrossArmDemo::getAppendSection() const
{
	if (m_ptrAppendSectionCurve.isValid())
		return m_ptrAppendSectionCurve->clone();
	else
		return nullptr;
}

void DemoObject::CrossArmDemo::setAppendSection(p3d::GeCurveArrayPtr ptrSectionCurve)
{
	if (ptrSectionCurve.isValid())
		m_ptrAppendSectionCurve = ptrSectionCurve->clone();
}

int DemoObject::CrossArmDemo::getOutWidth() const
{
	return m_nWidth;
}

void DemoObject::CrossArmDemo::setOutWidth(int nWidth)
{
	m_nWidth = nWidth;
}

bool DemoObject::CrossArmDemo::getDirection() const
{
	return m_bDirection;
}

void DemoObject::CrossArmDemo::setDirection(bool bDirection)
{
	m_bDirection = bDirection;
}

int DemoObject::CrossArmDemo::getHeight() const
{
	return m_nHeight;
}

void DemoObject::CrossArmDemo::setHeight(int nHeight)
{
	m_nHeight = nHeight;
}

int DemoObject::CrossArmDemo::getTopThickness() const
{
	return m_nTopThickness;
}

void DemoObject::CrossArmDemo::setTopThickness(int val)
{
	m_nTopThickness = val;
}

int DemoObject::CrossArmDemo::getBaseThickness() const
{
	return m_nBaseThickness;
}

void DemoObject::CrossArmDemo::setBaseThickness(int val)
{
	m_nBaseThickness = val;
}

int DemoObject::CrossArmDemo::getEdgeThickness() const
{
	return m_nEdgeThickness;
}

void DemoObject::CrossArmDemo::setEdgeThickness(int val)
{
	m_nEdgeThickness = val;
}

int DemoObject::CrossArmDemo::getNodeCount() const
{
	return m_nNode;
}

void DemoObject::CrossArmDemo::setNodeCount(int nCount)
{
	m_nNode = nCount;
}


::p3d::P3DStatus CrossArmDemo::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (T_Super::_copyToData(instance, project) != P3DStatus::SUCCESS)
		return ERROR;
	P3DStatus status;
	
	BPValue value;
	
	status = instance.setValue(Property_BaseWidth, BPValue(this->getOutWidth()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_Direction, BPValue(this->getDirection()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	
	status = instance.setValue(Property_Height, BPValue(this->getHeight()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_NodeCount, BPValue(this->getNodeCount()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_EdgeThickness, BPValue(this->getEdgeThickness()));
	if (P3DStatus::SUCCESS != status)
	return ERROR;

	status = instance.setValue(Property_BaseThickness, BPValue(this->getBaseThickness()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_TopThickness, BPValue(this->getTopThickness()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	return SUCCESS;
}

::p3d::P3DStatus CrossArmDemo::_initFromData(BIMBase::Core::BPDataCR instance)
{
	if (T_Super::_initFromData(instance) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = instance.getValue(value, Property_BaseWidth);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setOutWidth(value.getInteger());

	status = instance.getValue(value, Property_Direction);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setDirection(value.getBoolean());

	status = instance.getValue(value, Property_Height);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setHeight(value.getInteger());

	status = instance.getValue(value, Property_NodeCount);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setNodeCount(value.getInteger());

	status = instance.getValue(value, Property_EdgeThickness);
	if (P3DStatus::SUCCESS != status)
	return ERROR;
	setEdgeThickness(value.getInteger());

	status = instance.getValue(value, Property_BaseThickness);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setBaseThickness(value.getInteger());

	status = instance.getValue(value, Property_TopThickness);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setTopThickness(value.getInteger());

	return SUCCESS;
}


BIMBase::Core::BPGraphicsPtr DemoObject::CrossArmDemo::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool isDynamics)
{
	BPModelPtr ptrModel = project.getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;

	BPGraphicsPtr ptrPhysicalGeometry = ptrModel->createPhysicalGraphics();
	if (ptrPhysicalGeometry.isNull())
		return nullptr;

	int nDirection = m_bDirection ? -1 : 1;
	
	//绘制左边，右边通过镜像得到
	std::vector<std::vector<GePoint3d>> ptss;
	for (int i = 0; i < 4; i++)
	{
		std::vector<GePoint3d> pts;
		ptss.push_back(pts);
	}	
	double dxOffset = -m_nWidth / m_nNode;
	double dyOffsetTop = (m_nTopThickness - m_nEdgeThickness)/2 / m_nNode;
	double dyOffsetBase = (m_nBaseThickness - m_nEdgeThickness) / 2 / m_nNode;
	double dzOffset = m_nHeight*nDirection / m_nNode;
	GePoint3d pOffsetTop = GePoint3d::create(dxOffset, dyOffsetTop, dzOffset);
	GePoint3d pOffsetBase = GePoint3d::create(dxOffset, dyOffsetBase, 0);
	GePoint3d pointTop = GePoint3d::create(-m_nTopThickness / 2, -m_nTopThickness / 2, 0);
	GePoint3d pointBase = GePoint3d::create(-m_nBaseThickness / 2, -m_nBaseThickness / 2, m_nHeight*nDirection);
	GeTransform trans;
	trans.setByOriginAndScale(GePoint2d::createByZero(), 1, -1);
	for (int i = 0; i < m_nNode+1; i++)
	{
		ptss[0].push_back(pointTop + pOffsetTop*i);
		ptss[1].push_back(pointBase + pOffsetBase*i);
		ptss[2].push_back(GePoint3d::createByTransform(trans, ptss[0][i]));
		ptss[3].push_back(GePoint3d::createByTransform(trans, ptss[1][i]));
	}

	std::vector<GeTransform> vecTrans;
	for (int j = 1; j < 4; j++)
	{
		GeTransform transform = GeTransform::createByAxisAndRotationAngle(GeRay3d::createByOriginAndVector(GePoint3d::create(0, 0, 0), GeVec3d::create(0, 0, 1)), M_PI / 2 * j);
		vecTrans.push_back(transform);
	}

	trans.setByOriginAndScale(GePoint2d::createByZero(), -1, 1);

	//端部杆
	IGeSolidBasePtr ptrSolidD = createsolid(ptss[0][m_nNode], ptss[2][m_nNode], getMainSection());
	ptrPhysicalGeometry->addGeSolidBase(*ptrSolidD);
	IGeSolidBasePtr ptrSolidDT = ptrSolidD->deepClone();
	ptrSolidDT->transform(trans);
	ptrPhysicalGeometry->addGeSolidBase(*ptrSolidDT);

	
	for (int i = 0; i < 2; i++)
	{		
		GeCurveArrayPtr ptrCurveFront = getMainSection();
		GeCurveArrayPtr ptrCurveHor = getMainSection();
		GeCurveArrayPtr ptrCurveAppend = getAppendSection();
		if (i - 1 >= 0)
		{
			ptrCurveAppend->setByTransform(vecTrans[i]);
		}
		ptrCurveFront->setByTransform(vecTrans[0]);

		GeCurveArrayPtr ptrCurveBack = ptrCurveFront->clone();
		ptrCurveBack->setByTransform(vecTrans[0]);
		GeCurveArrayPtr ptrCurveAppendBack = ptrCurveAppend->clone();
		ptrCurveAppendBack->setByTransform(vecTrans[1]);

		GeCurveArrayPtr ptrCurveLeft = ptrCurveAppend->clone();
		ptrCurveLeft->setByTransform(vecTrans[0]);
		ptrCurveAppend->setByTransform(i == 0 ? GeTransform::create(0, 10, 0) : GeTransform::create(0, -10, 0));

		
		IGeSolidBasePtr ptrSolidH1 = createsolid(ptss[i][0], ptss[i][0] + GePoint3d::create(m_nTopThickness, 0, 0), ptrCurveHor);
		ptrPhysicalGeometry->addGeSolidBase(*ptrSolidH1);
		ptrCurveHor->setByTransform(vecTrans[2]);
		IGeSolidBasePtr ptrSolidH2 = createsolid(ptss[i + 2][0], ptss[i + 2][0] + GePoint3d::create(m_nBaseThickness, 0, 0), ptrCurveHor);
		ptrPhysicalGeometry->addGeSolidBase(*ptrSolidH2);


		for (int j = 0; j < ptss[0].size()-1; j++)
		{
			//主杆
			IGeSolidBasePtr ptrSolid = createsolid(ptss[i][j], ptss[i][j + 1], ptrCurveFront, true);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid);
			IGeSolidBasePtr ptrSolidT = ptrSolid->deepClone();
			ptrSolidT->transform(trans);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolidT);
			IGeSolidBasePtr ptrSolid2 = createsolid(ptss[i + 2][j], ptss[i + 2][j + 1], ptrCurveBack, true);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid2);
			IGeSolidBasePtr ptrSolid2T = ptrSolid2->deepClone();
			ptrSolid2T->transform(trans);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid2T);


			if (j == ptss[0].size() -2)
				continue;
			//竖向副杆
			IGeSolidBasePtr ptrSolid3 = createsolid(ptss[2 * i][j], ptss[2 * i + 1][j + 1], ptrCurveAppend);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid3);
			IGeSolidBasePtr ptrSolid3T = ptrSolid3->deepClone();
			ptrSolid3T->transform(trans);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid3T);
			IGeSolidBasePtr ptrSolid4 = createsolid(ptss[2 * i + 1][j + 1], ptss[2 * i][j + 1], ptrCurveAppend);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid4);
			IGeSolidBasePtr ptrSolid4T = ptrSolid4->deepClone();
			ptrSolid4T->transform(trans);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid4T);

			//横向副杆
			IGeSolidBasePtr ptrSolid5 = createsolid(ptss[i][j], ptss[i + 2][j + 1], ptrCurveAppend);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid5);
			IGeSolidBasePtr ptrSolid5T = ptrSolid5->deepClone();
			ptrSolid5T->transform(trans);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid5T);

			IGeSolidBasePtr ptrSolid6 = createsolid(ptss[i + 2][j], ptss[i][j + 1], ptrCurveLeft);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid6);
			IGeSolidBasePtr ptrSolid6T = ptrSolid6->deepClone();
			ptrSolid6T->transform(trans);
			ptrPhysicalGeometry->addGeSolidBase(*ptrSolid6T);

		}
	}

	return ptrPhysicalGeometry;
}

IGeSolidBasePtr DemoObject::CrossArmDemo::createsolid(GePoint3d sPoint, GePoint3d ePoint, GeCurveArrayPtr sectionCurve, bool bAdjustPoint)
{
	GePoint3d _sPoint = sPoint, _ePoint = ePoint;
	if (bAdjustPoint)
	{
		_sPoint.y = 0;
		_ePoint.y = 0;
	}	

	GeRotMatrix matrix = GeRotMatrix::createIdentityMatrix();
	matrix.setByRotationFromVectorToVector(GeVec3d::create(0, 0, 1), GeVec3d::createByStartEnd(_sPoint, _ePoint));
	GeCurveArrayPtr setctionA = sectionCurve->clone();
	GeTransform transA = GeTransform::createByProduct(GeTransform::create(sPoint), GeTransform::createByMatrixAndFixedPoint(matrix, GePoint3d::create(0,0,0)));
	setctionA->setByTransform(transA);

	GeCurveArrayPtr setctionB = sectionCurve->clone();
	GeTransform transB = GeTransform::createByProduct(GeTransform::create(ePoint), GeTransform::createByMatrixAndFixedPoint(matrix, GePoint3d::create(0, 0, 0)));
	setctionB->setByTransform(transB);

	GeRuledSweepInfo ruledSweepInfo(setctionA, setctionB, true);
	return IGeSolidBase::createGeRuledSweep(ruledSweepInfo);
}

GeCurveArrayPtr DemoObject::CrossArmDemo::initSectionBase(int nWidth)
{
	pvector<GePoint3d> points;
	points.push_back(GePoint3d::create(0, 0, 0));
	points.push_back(GePoint3d::create(nWidth, 0, 0));
	points.push_back(GePoint3d::create(nWidth, 10, 0));
	points.push_back(GePoint3d::create(10, 10, 0));
	points.push_back(GePoint3d::create(10, nWidth, 0));
	points.push_back(GePoint3d::create(0, nWidth, 0));
	return GeCurveArray::createLinestringArray(points, GeCurveArray::BOUNDARY_TYPE_Outer);
}

