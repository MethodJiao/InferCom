#include "stdafx.h"
#include "CubeDemo.h"

#define Property_Lenght                          "Length"
#define Property_Width                           "Width"
#define Property_Height                          "Height"

using namespace DemoObject;

CubeDemo::CubeDemo()
{
	m_nLenght = 1000;
	m_nWidth = 200;
	m_nHeight = 3000;
}

CubeDemo::~CubeDemo()
{

}

int CubeDemo::getWidth() const
{
	return m_nWidth;
}

void CubeDemo::setWidth(int nWidth)
{
	m_nWidth = nWidth;
}

int CubeDemo::getLength() const
{
	return m_nLenght;
}

void CubeDemo::setLength(int nLength)
{
	m_nLenght = nLength;
}

int CubeDemo::getHeight() const
{
	return m_nHeight;
}

void CubeDemo::setHeight(int nHeight)
{
	m_nHeight = nHeight;
}

::p3d::P3DStatus CubeDemo::_copyToData(BIMBase::Core::BPDataR instance, BIMBase::Core::BPProject& project) const
{
	if (T_Super::_copyToData(instance, project) != P3DStatus::SUCCESS)
		return ERROR;

	P3DStatus status;
	status = instance.setValue(Property_Lenght, BPValue(this->getLength()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_Width, BPValue(this->getWidth()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = instance.setValue(Property_Height, BPValue(this->getHeight()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	return SUCCESS;
}

::p3d::P3DStatus CubeDemo::_initFromData(BIMBase::Core::BPDataCR instance)
{
	if (T_Super::_initFromData(instance) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = instance.getValue(value, Property_Lenght);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setLength(value.getInteger());

	status = instance.getValue(value, Property_Width);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setWidth(value.getInteger());

	status = instance.getValue(value, Property_Height);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setHeight(value.getInteger());

	return SUCCESS;
}

BIMBase::Core::BPGraphicsPtr DemoObject::CubeDemo::createGraphicsPlane(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId)
{
	BPModelPtr ptrModel = project.getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;

	BPGraphicsPtr ptrPhysicalGeometry = ptrModel->createPhysicalGraphics();
	if (ptrPhysicalGeometry.isNull())
		return nullptr;

	int nWidth = getWidth();
	int nLength = getLength();
	//绘制底部外轮廓
	GeCurveArrayPtr ptrOutLines = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	pvector<GePoint3d> pts;
	IGeCurveBasePtr pLine;

	pts.push_back(GePoint3d::create(0, -nWidth / 2, 0));
	pts.push_back(GePoint3d::create(nLength, -nWidth / 2, 0));
	pts.push_back(GePoint3d::create(nLength, nWidth / 2, 0));
	pts.push_back(GePoint3d::create(0, nWidth / 2, 0));
	pts.push_back(GePoint3d::create(0, -nWidth / 2, 0));
	pLine = IGeCurveBase::createLineString(pts);
	ptrOutLines->push_back(pLine);
	ptrPhysicalGeometry->addGeCurveArray(*ptrOutLines);
	return ptrPhysicalGeometry;
}
BIMBase::Core::BPGraphicsPtr DemoObject::CubeDemo::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics)
{

	BPModelPtr ptrModel = project.getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;

	BPGraphicsPtr ptrGraphic = ptrModel->createPhysicalGraphics();
	if (ptrGraphic.isNull())
		return nullptr;

	int nWidth = getWidth();
	int nLength = getLength() ;
	int nHeight = getHeight();

	//绘制底部外轮廓
	GeCurveArrayPtr ptrOutLines = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	pvector<GePoint3d> pts;
	IGeCurveBasePtr ptrLine;

	pts.push_back(GePoint3d::create(0, -nWidth / 2, 0));
	pts.push_back(GePoint3d::create(nLength, -nWidth / 2, 0));
	pts.push_back(GePoint3d::create(nLength, nWidth / 2, 0));
	pts.push_back(GePoint3d::create(0, nWidth / 2, 0));
	pts.push_back(GePoint3d::create(0, -nWidth / 2, 0));
	ptrLine = IGeCurveBase::createLineString(pts);
	ptrOutLines->push_back(ptrLine);

	//向Z方向拉伸
	GeVec3d veccc = GeVec3d::create(0, 0, nHeight);
	GeExtrusionInfo extrData(ptrOutLines, veccc, true);
	IGeSolidBasePtr ptrExtrusion = IGeSolidBase::createGeExtrusion(extrData);	

	ptrGraphic->addGeSolidBase(*ptrExtrusion);

	//---------------以下为创建洞口代码，非常规造型代码----------------------
	if (getData(project) == nullptr)
		return ptrGraphic;

	//获取cube关联的openning对象，用于布尔计算
	BPDataKeyArray relDataIds;
	BPRelationshipFinder::getRelatedDatasByRelationship(relDataIds, getDataKey(), project, PBM_SCHEMA_Demo, PBM_RELSHIP_CUBEWITHOPENNING);
	
	if (relDataIds.size() == 0)
		return ptrGraphic;
	
	BPGraphicsPtr ptrGraphicResult = ptrModel->createPhysicalGraphics();
	if (ptrGraphicResult.isNull())
		return nullptr;

	for each (auto dataKey in relDataIds)
	{
		BPDataPtr ptrData = BPDataUtil::getDataByKey(dataKey, project);
		if (ptrData == nullptr)
			continue;
		OpenningDemoPtr ptrOpenning = OpenningDemo::create(*ptrData);
		if (ptrOpenning == nullptr)
			continue;

		//如果当前宽度不一致，修改洞口宽度
		if (abs(ptrOpenning->getWidth() - getWidth()) > 0.01)
		{
			ptrOpenning->setWidth(getWidth());
			ptrOpenning->replaceInProject(project);
		}

		//获取洞口图素，返回局部坐标系下图素
		BPGraphicsPtr ptrGraphicOpenning = ptrOpenning->createPhysicalGraphics(project, ptrModel->getModelId(), true);
		GeTransform openningTrans = ptrOpenning->getTransform();
		GeTransform cubeTrans = getTransform();
		cubeTrans.setByInverse(cubeTrans);

		BPGraphicsPtr ptrGraphicResultTemp = ptrModel->createPhysicalGraphics();
		if (ptrGraphicResultTemp.isNull())
			continue;

		//当前cube为局部坐标系下图素，需要将openning的图素转到世界坐标系下，再转到cube的局部坐标系下进行bool
		BPGraphicsUtils::transformPhysicalGraphics(*ptrGraphicOpenning, GeTransform::createByProduct(cubeTrans, openningTrans));
		BPSolidBooleanUtil::doBoolean(ptrGraphicResultTemp, ptrGraphic, ptrGraphicOpenning, BPBooleanOp::Substract);
		ptrGraphic = ptrModel->createPhysicalGraphics();
		if (ptrGraphic.isNull())
			continue;
		BPGraphicsUtils::copyPhysicalGraphics(*ptrGraphic, *ptrGraphicResultTemp);
		ptrGraphicResult = ptrGraphicResultTemp;
	}
	return ptrGraphicResult;
}

BIMBase::BPSnapStatus DemoObject::CubeDemo::_onSnap(BIMBase::Core::BPSnapContext& snapContext)
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

GePoint3d DemoObject::CubeDemo::getStartPoint() const
{
	//通过转换矩阵获取原点
	return getPlacement().getOrigin();
}

GePoint3d DemoObject::CubeDemo::getMiddlePoint() const
{
	GePoint3d ptMiddle = GePoint3d::create(getLength() / 2.0, 0, 0);
	//墙局部坐标系转到世界坐标系下
	GeTransform trans = getPlacement().toTransform();
	trans.multiply(ptMiddle);
	return ptMiddle;
}

GePoint3d DemoObject::CubeDemo::getEndPoint() const
{
	GePoint3d ptEnd = GePoint3d::create(getLength(), 0, 0);
	//墙局部坐标系转到世界坐标系下
	GeTransform trans = getPlacement().toTransform();
	trans.multiply(ptEnd);
	return ptEnd;
}

