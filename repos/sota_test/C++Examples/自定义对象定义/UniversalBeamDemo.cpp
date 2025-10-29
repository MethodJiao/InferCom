#include "stdafx.h"
#include "UniversalBeamDemo.h"
using namespace std;

const std::string PROPERTY_LENGTH = "Length";
const std::string PROPERTY_WIDTH = "Width";
const std::string PROPERTY_THICK = "Thick";
const std::string PROPERTY_INNER_HEIGHT = "InnerHeight";

#define Property_DemoArray                          "DemoArray"

using namespace DemoObject;

UniversalBeamDemo::UniversalBeamDemo()
{
	m_nThick = 3000;
	m_nWidth = 1500;
	m_nLenght = 2000;
	m_nInnerHeight = 300;
}

UniversalBeamDemo::~UniversalBeamDemo()
{

}

int UniversalBeamDemo::getWidth() const
{
	return m_nWidth;
}

void UniversalBeamDemo::setWidth(int nWidth)
{
	m_nWidth = nWidth;
}

int UniversalBeamDemo::getLength() const
{
	return m_nLenght;
}

void UniversalBeamDemo::setLength(int nLength)
{
	m_nLenght = nLength;
}

int UniversalBeamDemo::getThick() const
{
	return m_nThick;
}

void UniversalBeamDemo::setThick(int nThick)
{
	m_nThick = nThick;
}

int UniversalBeamDemo::getInnerHeight() const
{
	return m_nInnerHeight;
}

void   UniversalBeamDemo::setInnerHeight(int nInnerHeight)
{
	m_nInnerHeight = nInnerHeight;
}

::p3d::P3DStatus UniversalBeamDemo::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (T_Super::_copyToData(instance, project) != P3DStatus::SUCCESS)
		return ERROR;

	P3DStatus status;
	status = instance.setValue(PROPERTY_LENGTH.c_str(), BPValue(this->getLength()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_WIDTH.c_str(), BPValue(this->getWidth()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_THICK.c_str(), BPValue(this->getThick()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(PROPERTY_INNER_HEIGHT.c_str(), BPValue(this->getInnerHeight()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;



	return SUCCESS;
}

::p3d::P3DStatus UniversalBeamDemo::_initFromData(BIMBase::Core::BPDataCR instance)
{
	if (T_Super::_initFromData(instance) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = instance.getValue(value, PROPERTY_LENGTH.c_str());
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setLength(value.getInteger());

	status = instance.getValue(value, PROPERTY_WIDTH.c_str());
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setWidth(value.getInteger());

	status = instance.getValue(value, PROPERTY_THICK.c_str());
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setThick(value.getInteger());

	status = instance.getValue(value, PROPERTY_INNER_HEIGHT.c_str());
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setInnerHeight(value.getInteger());

	return SUCCESS;
}

BIMBase::Core::BPGraphicsPtr DemoObject::UniversalBeamDemo::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics)
{
	BPModelPtr ptrModel = project.getModelById(modelId);
	if (ptrModel.isNull())
	{
		return nullptr;
	}

	BPGraphicsPtr ptrGraphic = ptrModel->createPhysicalGraphics();
	if (ptrGraphic.isNull())
	{
		return nullptr;
	}

	double L = m_nLenght / 2.0;
	double W = m_nWidth / 2.0;
	double In = m_nInnerHeight / 2.0;

	//绘制底部外轮廓
	GeCurveArrayPtr ptrOutLines = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	pvector<GePoint3d> pts;
	pts.push_back({ L ,W ,0 });//0
	pts.push_back({ L,W - m_nInnerHeight ,0 });//1
	pts.push_back({ In,W - m_nInnerHeight ,0 });//2
	pts.push_back({ In,  m_nInnerHeight - W ,0 });//3
	pts.push_back({ L,m_nInnerHeight - W ,0 });//4
	pts.push_back({ L ,-W ,0 });//5
	pts.push_back({ -L ,-W ,0 });//6
	pts.push_back({ -L,m_nInnerHeight - W,0 });//7
	pts.push_back({ -In,  m_nInnerHeight - W ,0 });//8
	pts.push_back({ -In,W - m_nInnerHeight ,0 });//9
	pts.push_back({ -L,W - m_nInnerHeight ,0 });//10
	pts.push_back({ -L ,W ,0 });//11
	pts.push_back({ L ,W ,0 });//12
	IGeCurveBasePtr ptrLine;

	ptrLine = IGeCurveBase::createLineString(pts);
	ptrOutLines->push_back(ptrLine);

	//向Z方向拉伸
	GeVec3d vecZ = GeVec3d::create(0, 0, m_nThick);
	GeExtrusionInfo extrData(ptrOutLines, vecZ, true);
	IGeSolidBasePtr ptrExtrusion = IGeSolidBase::createGeExtrusion(extrData);

	ptrGraphic->addGeSolidBase(*ptrExtrusion);
	return ptrGraphic;
}

BIMBase::BPSnapStatus DemoObject::UniversalBeamDemo::_onSnap(BIMBase::Core::BPSnapContext& snapContext)
{
	BPSnapMode snapMode = snapContext.getSnapMode();

	p3d::platform::BPSnapPathImpP pSnapPath = snapContext.getSnapPath();
	if (NULL == pSnapPath)
		return BPSnapStatus::eDisabled;

	//forceHot为false时，只有当用户将鼠标靠近捕捉点的位置，才会捕捉到该点；
	//forceHor为true时，不论鼠标与捕捉点的距离多远，都能捕捉到该点。
	bool forceHot = false;
	GePoint3d hitPoint;

	switch (snapMode)
	{
	case BPSnapMode::MidPoint:
	{
		//获取长度方向的中点
		GePoint3d ptMiddle = GePoint3d::create(getLength() / 2.0, 0, 0);
		//局部坐标系转到世界坐标系下
		GeTransform trans = getPlacement().toTransform();
		trans.multiply(ptMiddle);
		hitPoint = ptMiddle;

		//捕捉点设置
		snapContext.setSnapInfo(0, snapMode, hitPoint, forceHot, true, 0, NULL);
		return BPSnapStatus::eSuccess;
	}
	break;
	}

	return BPSnapStatus::eNotSnappable;
}

GePoint3d DemoObject::UniversalBeamDemo::getCenterPoint() const
{
	//通过转换矩阵获取原点
	return getPlacement().getOrigin();
}

GePoint3d DemoObject::UniversalBeamDemo::getStartPoint() const
{
	double L = m_nLenght / 2.0;
	double W = m_nWidth / 2.0;
	GePoint3d ptStart{ -L, -W, 0 };
	GeTransform trans = getPlacement().toTransform();
	trans.multiply(ptStart);
	return ptStart;
}

GePoint3d DemoObject::UniversalBeamDemo::getEndPoint() const
{
	double L = m_nLenght / 2.0;
	double W = m_nWidth / 2.0;
	GePoint3d ptEnd{ L, -W, 0 };
	GeTransform trans = getPlacement().toTransform();
	trans.multiply(ptEnd);
	return ptEnd;
}

GePoint3d    DemoObject::UniversalBeamDemo::getUpStartPoint() const
{
	double L = m_nLenght / 2.0;
	double W = m_nWidth / 2.0;
	GePoint3d ptUpStar{ L,m_nInnerHeight - W,0 };
	GeTransform trans = getPlacement().toTransform();
	trans.multiply(ptUpStar);
	return ptUpStar;
}

GePoint3d    DemoObject::UniversalBeamDemo::getUpEndPoint() const
{
	double L = m_nLenght / 2.0;
	double W = m_nWidth / 2.0;
	GePoint3d ptUpEnd{ -L,m_nInnerHeight - W,0 };
	GeTransform trans = getPlacement().toTransform();
	trans.multiply(ptUpEnd);
	return ptUpEnd;
}