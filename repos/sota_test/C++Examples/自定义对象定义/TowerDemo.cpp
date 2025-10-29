#include "stdafx.h"



#define Property_MainSection                        "MainSectionCurve"
#define Property_AppendSection                      "AppendSectionCurve"
#define Property_BaseWidth                          "BaseWidth"
#define Property_Slope                              "Slope"
#define Property_Height                             "Height"
#define Property_NodeCount                          "NodeCount"
#define Property_CrossArms                          "CrossArms"

using namespace DemoObject;
using namespace p3d;

TowerDemo::TowerDemo()
	: m_nWidth(2000)
	, m_dSlope(1.56)
	, m_nHeight(20000)
	, m_nNode(20)
{
	m_ptrMainSectionCurve = initSectionBase(80);
	m_ptrAppendSectionCurve = initSectionBase(40);
	initPoints();
	m_vctArmPara.push_back(make_pair(19, -2000));
	m_vctArmPara.push_back(make_pair(19, 3000));
	m_vctArmPara.push_back(make_pair(16, 3000));
	m_vctArmPara.push_back(make_pair(13, 4000));
	for (int i = 0; i < m_vctArmPara.size(); i++)
	{
		CrossArmDemoPtr ptrArm = new CrossArmDemo();
		int index2 = m_vctArmPara[i].second > 0 ? m_vctArmPara[i].first - 1 : m_vctArmPara[i].first + 1;
		GePoint3d point1, point2;
		double width1 = 0, width2 = 0;
		getNodeParameter(index2, point1, width1);
		getNodeParameter(m_vctArmPara[i].first, point2, width2);
		ptrArm->setTopThickness(width2);
		ptrArm->setBaseThickness(width1);
		ptrArm->setOutWidth(abs(m_vctArmPara[i].second));
		ptrArm->setHeight(m_vctArmPara[i].second > 0 ? (point2.z - point1.z) : (point1.z - point2.z));
		ptrArm->setDirection(m_vctArmPara[i].second > 0);
		GeTransform trans = GeTransform::create(point2 + GePoint3d::create(width2 / 2, width2 / 2, 0));
		BPPlacement palcement;
		palcement.fromTransform(trans);
		ptrArm->setPlacement(palcement);
		m_vctCrossArm.push_back(ptrArm);
	}
}

TowerDemo::~TowerDemo()
{

}

bool DemoObject::TowerDemo::getNodeParameter(IN int nIndex, OUT GePoint3dR nodePoint, OUT double& width)
{
	if (m_points.size() == 0)
		return false;
	if (m_points[0].size() <= nIndex)
		return false;
	nodePoint = m_points[0][nIndex];
	width = nodePoint.distance(m_points[1][nIndex]);
	return true;
}

p3d::GeCurveArrayPtr DemoObject::TowerDemo::getMainSection() const
{
	if (m_ptrMainSectionCurve.isValid())
		return m_ptrMainSectionCurve->clone();
	else
		return nullptr;
}

void DemoObject::TowerDemo::setMainSection(p3d::GeCurveArrayPtr sectionCurve)
{
	if (sectionCurve.isValid())
		m_ptrMainSectionCurve = sectionCurve->clone();
}

p3d::GeCurveArrayPtr DemoObject::TowerDemo::getAppendSection() const
{
	if (m_ptrAppendSectionCurve.isValid())
		return m_ptrAppendSectionCurve->clone();
	else
		return nullptr;
}

void DemoObject::TowerDemo::setAppendSection(p3d::GeCurveArrayPtr ptrSectionCurve)
{
	if (ptrSectionCurve.isValid())
		m_ptrAppendSectionCurve = ptrSectionCurve->clone();
}

int DemoObject::TowerDemo::getBaseWidth() const
{
	return m_nWidth;
}

void DemoObject::TowerDemo::setBaseWidth(int nWidth)
{
	m_nWidth = nWidth;
}

double DemoObject::TowerDemo::getSlope() const
{
	return m_dSlope;
}

void DemoObject::TowerDemo::setSlope(double dSlope)
{
	m_dSlope = dSlope;
}

int DemoObject::TowerDemo::getHeight() const
{
	return m_nHeight;
}

void DemoObject::TowerDemo::setHeight(int nHeight)
{
	m_nHeight = nHeight;
}

int DemoObject::TowerDemo::getNodeCount() const
{
	return m_nNode;
}

void DemoObject::TowerDemo::setNodeCount(int nCount)
{
	m_nNode = nCount;
}


::p3d::P3DStatus TowerDemo::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (T_Super::_copyToData(instance, project) != P3DStatus::SUCCESS)
		return ERROR;

	
	BPValue value;
	p3d::P3DStatus status2 = BPDataUtil::setCurveVectorToECValue(value, this->getMainSection());
	if (SUCCESS != status2)
		return ERROR;

	status2 = BPDataUtil::setCurveVectorToECValue(value, this->getAppendSection());
	if (SUCCESS != status2)
		return ERROR;

	P3DStatus status;
	status = instance.setValue(Property_BaseWidth, BPValue(this->getBaseWidth()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	
	status = instance.setValue(Property_Slope, BPValue(this->getSlope()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_Height, BPValue(this->getHeight()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_NodeCount, BPValue(this->getNodeCount()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.addArrayElements(Property_CrossArms, m_vctCrossArm.size());
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	for (int i = 0; i < m_vctCrossArm.size(); i++)
	{
		BPValue _value;
		BPDataPtr _ptrInstance = m_vctCrossArm.at(i)->getData(project);
		if (_ptrInstance.isNull())
		{
			_ptrInstance = BPDataUtil::createDataByName(project, PBM_SCHEMA_Demo, PBM_CLASS_CROSSARM_Demo, BPDataFlag::enNormal);
			if (_ptrInstance.isNull())
				continue;
			m_vctCrossArm.at(i)->copyToData(*_ptrInstance, project);
		}

		if (_ptrInstance == nullptr)
			return ERROR;
		_value.setStruct(_ptrInstance.get());
		status = instance.setValue(Property_CrossArms, _value, i);
		if (P3DStatus::SUCCESS != status)
			return ERROR;
	}

	return SUCCESS;
}

::p3d::P3DStatus TowerDemo::_initFromData(BIMBase::Core::BPDataCR instance)
{
	if (T_Super::_initFromData(instance) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = instance.getValue(value, Property_MainSection);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	m_ptrMainSectionCurve->clear();
	BIMBase::Core::BPDataUtil::getCurveVectorFromECValue(m_ptrMainSectionCurve, value);

	status = instance.getValue(value, Property_AppendSection);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	m_ptrAppendSectionCurve->clear();
	BIMBase::Core::BPDataUtil::getCurveVectorFromECValue(m_ptrAppendSectionCurve, value);

	status = instance.getValue(value, Property_BaseWidth);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setBaseWidth(value.getInteger());

	status = instance.getValue(value, Property_Slope);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setSlope(value.getDouble());

	status = instance.getValue(value, Property_Height);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setHeight(value.getInteger());

	status = instance.getValue(value, Property_NodeCount);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setNodeCount(value.getInteger());

	m_vctCrossArm.clear();
	status = instance.getValue(value, Property_CrossArms);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	int arrCount = value.getArrayCount();
	for (int i = 0; i < arrCount; i++)
	{
		BPValue _value;

		status = instance.getValue(_value, Property_CrossArms, i);
		if (P3DStatus::SUCCESS != status)
			return ERROR;
		BPDataPtr insPtr = _value.getStruct();

		if (insPtr == nullptr)
			return ERROR;
		CrossArmDemoPtr ptrArm = new CrossArmDemo();
		P3DStatus _status = ptrArm->initFromData(*insPtr);
		if (0 != _status)
			return ERROR;
		m_vctCrossArm.push_back(ptrArm);
	}

	return SUCCESS;
}


BIMBase::Core::BPGraphicsPtr DemoObject::TowerDemo::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool isDynamics)
{
	BPModelPtr ptrModel = project.getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;

	BPGraphicsPtr ptrGraphic = ptrModel->createPhysicalGraphics();
	if (ptrGraphic.isNull())
		return nullptr;
	
	
	//»æÖÆÖ÷¸Ë

	std::vector<GeTransform> vecTrans;
	for (int j = 1; j < 4; j++)
	{
		GeTransform transform = GeTransform::createByAxisAndRotationAngle(GeRay3d::createByOriginAndVector(GePoint3d::create(0, 0, 0), GeVec3d::create(0, 0, 1)), M_PI / 2 * j);
		vecTrans.push_back(transform);
	}
	
	for (int i = 0; i < m_points.size(); i++)
	{		
		GeCurveArrayPtr ptrCurveBase = getMainSection();
		GeCurveArrayPtr ptrCurveAppend = getAppendSection();
		if (i - 1 >= 0)
		{
			ptrCurveBase->setByTransform(vecTrans[i - 1]);
			ptrCurveAppend->setByTransform(vecTrans[i - 1]);
		}
		GeCurveArrayPtr ptrCurveLeft = ptrCurveAppend->clone();
		ptrCurveLeft->setByTransform(vecTrans[0]);
		ptrCurveAppend->setByTransform(GeTransform::create(0, 10, 0));
		for (int j = 0; j < m_points[0].size() - 1; j++)
		{
			IGeSolidBasePtr ptrSolid = createsolid(m_points[i][j], m_points[i][j + 1], ptrCurveBase);
			ptrGraphic->addGeSolidBase(*ptrSolid);
			IGeSolidBasePtr ptrSolid1 = createsolid(m_points[i][j], m_points[(i + 1) % 4][j + 1], ptrCurveAppend, true);
			ptrGraphic->addGeSolidBase(*ptrSolid1);
			IGeSolidBasePtr ptrSolid2 = createsolid(m_points[i][j], m_points[(i + 3) % 4][j + 1], ptrCurveLeft, true);
			ptrGraphic->addGeSolidBase(*ptrSolid2);
		}
	}

	//ºáµ£
	vector<BPGraphicsPtr> vctGraphic;
	for (int i = 0; i < m_vctCrossArm.size(); i++)
	{
		BPGraphicsPtr ptrGraphicArm = m_vctCrossArm[i]->createPhysicalGraphics(project, modelId, isDynamics);
		GeTransform trans = m_vctCrossArm[i]->getTransform();
		BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphicArm, trans);
		vctGraphic.push_back(ptrGraphicArm);
	}
	BPGraphicsUtils::insertGraphics(ptrGraphic, vctGraphic);

	return ptrGraphic;
}

IGeSolidBasePtr DemoObject::TowerDemo::createsolid(GePoint3d sPoint, GePoint3d ePoint, GeCurveArrayPtr sectionCurve, bool bAdjustPoint)
{
	if (bAdjustPoint)
	{
		GeVec3d adjustDirection = GeVec3d::createByStartEnd(sPoint, ePoint);
		adjustDirection.z = 0;
		adjustDirection.normalize();
		adjustDirection = adjustDirection * 30;

		sPoint = sPoint + adjustDirection;
		ePoint = ePoint - adjustDirection;
	}
	

	GeRotMatrix matrix = GeRotMatrix::createIdentityMatrix();
	matrix.setByRotationFromVectorToVector(GeVec3d::create(0, 0, 1), GeVec3d::createByStartEnd(sPoint, ePoint));
	GeCurveArrayPtr ptrSetctionA = sectionCurve->clone();
	GeTransform transA = GeTransform::createByProduct(GeTransform::create(sPoint), GeTransform::createByMatrixAndFixedPoint(matrix, GePoint3d::create(0,0,0)));
	ptrSetctionA->setByTransform(transA);

	GeCurveArrayPtr setctionB = sectionCurve->clone();
	GeTransform transB = GeTransform::createByProduct(GeTransform::create(ePoint), GeTransform::createByMatrixAndFixedPoint(matrix, GePoint3d::create(0, 0, 0)));
	setctionB->setByTransform(transB);

	GeRuledSweepInfo ruledSweepInfo(ptrSetctionA, setctionB, true);
	return IGeSolidBase::createGeRuledSweep(ruledSweepInfo);
}

GeCurveArrayPtr DemoObject::TowerDemo::initSectionBase(int nWidth)
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

void DemoObject::TowerDemo::initPoints()
{
	m_points.clear();
	for (int i = 0; i < 4; i++)
	{
		std::vector<GePoint3d> pts;
		m_points.push_back(pts);
	}
	
	double xOffset = m_nHeight / tan(m_dSlope) / 1.414 / m_nNode;
	double zOffset = m_nHeight / m_nNode;
	GePoint3d pointBase1 = GePoint3d::create(-m_nWidth / 2, -m_nWidth / 2, 0);
	std::vector<GeTransform> vecTrans;
	for (int j = 1; j < 4; j++)
	{
		GeTransform transform = GeTransform::createByAxisAndRotationAngle(GeRay3d::createByOriginAndVector(GePoint3d::create(0, 0, 0), GeVec3d::create(0, 0, 1)), M_PI / 2 * j);
		vecTrans.push_back(transform);
	}

	for (int i = 0; i < m_nNode + 1; i++)
	{
		GePoint3d point = GePoint3d::create(-m_nWidth / 2 + xOffset * i, -m_nWidth / 2 + xOffset * i, zOffset * i);
		m_points[0].push_back(point);
		for (int j = 1; j < 4; j++)
		{
			GePoint3d pointtt = GePoint3d::createByTransform(vecTrans[j - 1], point);
			m_points[j].push_back(pointtt);
		}
	}
}

static void LayoutTower()
{
	TowerDemoPtr ptrTower = TowerDemo::create();
	BIMBase::Core::BPProjectP pProject = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == NULL)
		return;

	::BIMBase::PModelId modelId = pProject->getActiveModel()->getModelId();

	ptrTower->addToProject(*pProject, modelId);
}


AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(_T("layoutTowerDemo"), LayoutTower);
AutoDoRegisterFunctionsEnd
