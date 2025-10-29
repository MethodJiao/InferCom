#include "stdafx.h"
#include "ModeArchWallDemo.h"

#define Property_Lenght                          "Length"
#define Property_Width                           "Width"
#define Property_Height                          "Height"

using namespace DemoObject;

ModeArchWallDemo::ModeArchWallDemo()
{
}

ModeArchWallDemo::~ModeArchWallDemo()
{

}

BIMBase::Core::BPGraphicsPtr DemoObject::ModeArchWallDemo::_createPhysicalGraphics(BIMBase::Core::BPProjectR project, BIMBase::PModelIdCR modelId, bool bIsDynamics)
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

	ptrGraphic->addGeSolidBase(*ptrExtrusion);

	return ptrGraphic;
}