#include "stdafx.h"


#define Property_Lenght                          "Length"
#define Property_Width                           "Width"
#define Property_Height                          "Height"


using namespace DemoObject;

OpenningDemo::OpenningDemo()
{
	m_nLenght = 500;
	m_nWidth = 200;
	m_nHeight = 500;
}

OpenningDemo::~OpenningDemo()
{

}


int OpenningDemo::getWidth() const
{
	return m_nWidth;
}

void OpenningDemo::setWidth(int nWidth)
{
	m_nWidth = nWidth;
}

int OpenningDemo::getLength() const
{
	return m_nLenght;
}

void OpenningDemo::setLength(int nLength)
{
	m_nLenght = nLength;
}

int OpenningDemo::getHeight() const
{
	return m_nHeight;
}

void OpenningDemo::setHeight(int nHeight)
{
	m_nHeight = nHeight;
}

::p3d::P3DStatus      OpenningDemo::_copyToData(::BIMBase::Core::BPDataR data, ::BIMBase::Core::BPProjectR project) const
{
	if (T_Super::_copyToData(data, project) != P3DStatus::SUCCESS)
		return ERROR;

	P3DStatus status;
	status = data.setValue(Property_Lenght, BPValue(this->getLength()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = data.setValue(Property_Width, BPValue(this->getWidth()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	status = data.setValue(Property_Height, BPValue(this->getHeight()));
	if (P3DStatus::SUCCESS != status)
		return ERROR;

	return SUCCESS;
}

::p3d::P3DStatus  OpenningDemo::_initFromData(::BIMBase::Core::BPDataCR data)
{
	if (T_Super::_initFromData(data) != P3DStatus::SUCCESS)
		return ERROR;

	BPValue value;
	P3DStatus status;

	status = data.getValue(value, Property_Lenght);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setLength(value.getInteger());

	status = data.getValue(value, Property_Width);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setWidth(value.getInteger());

	status = data.getValue(value, Property_Height);
	if (P3DStatus::SUCCESS != status)
		return ERROR;
	setHeight(value.getInteger());


	return SUCCESS;
}

BIMBase::Core::BPGraphicsPtr    OpenningDemo::_createPhysicalGraphics(::BIMBase::Core::BPProjectR project, ::BIMBase::PModelIdCR modelId, bool isDynamics)
{
	BPModelPtr ptrModel = project.getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;

	BPGraphicsPtr ptrGraphic = ptrModel->createPhysicalGraphics();
	if (ptrGraphic.isNull())
		return nullptr;

	int nWidth = getWidth();
	int nLength = getLength();
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

	//颜色   shicheng
	BPSymbology symb;
	symb.style = 0;  //线型
	symb.weight = 0;  //线宽
	symb.color = BPColorUtil::getEntityColor(RGB(255, 255, 255), project, true);

	ptrGraphic->addGeSolidBase(*ptrExtrusion, symb, 1);

	return ptrGraphic;
}
