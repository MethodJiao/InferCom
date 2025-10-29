#include "stdafx.h"
#include "LineDemo.h"


using namespace DemoObject;

LineDemo::LineDemo()
{

}

LineDemo::~LineDemo()
{

}

BIMBase::Core::BPGraphicsPtr DemoObject::LineDemo::createGraphics()
{
	// 获取当前工程
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return nullptr;
	PModelId modelId = pProject->getDefaultModelId();
	BPModelPtr ptrModel = pProject->getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;
	if (m_ptrLineGraphics != nullptr)
		return m_ptrLineGraphics;
	BPGraphicsPtr ptrPhysicalGeometry = ptrModel->createPhysicalGraphics();
	if (ptrPhysicalGeometry.isNull())
		return nullptr;
	GeSegment3d seg = GeSegment3d::create(GePoint3d::create(4000, 0, 0), GePoint3d::create(5000, 0, 0));
	IGeCurveBasePtr ptrCurve = IGeCurveBase::createSegment(seg);
	ptrPhysicalGeometry->addGeCurve(*ptrCurve);
	
	return ptrPhysicalGeometry;
}

SoildCubeDemo::SoildCubeDemo()
{

}

SoildCubeDemo::~SoildCubeDemo()
{

}


BIMBase::Core::BPGraphicsPtr DemoObject::SoildCubeDemo::createGraphics()
{
	// 获取当前工程
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return nullptr;
	PModelId modelId = pProject->getDefaultModelId();
	BPModelPtr ptrModel = pProject->getModelById(modelId);
	if (ptrModel.isNull())
		return nullptr;
	if (m_ptrSoildGraphics != nullptr)
		return m_ptrSoildGraphics;

	BPGraphicsPtr ptrPhysicalGeometry = ptrModel->createPhysicalGraphics();
	if (ptrPhysicalGeometry.isNull())
		return nullptr;
	
	GeCurveArrayPtr parityRegion = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_ParityRegion);
	GeCurveArrayPtr outLines = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer);
	pvector<GePoint3d> pts;
	IGeCurveBasePtr pLine;
	//示意个立方体
	pts.push_back(GePoint3d::create(0, -1000 / 2, 0));
	pts.push_back(GePoint3d::create(2000, -1000 / 2, 0));
	pts.push_back(GePoint3d::create(2000, 1000 / 2, 0));
	pts.push_back(GePoint3d::create(0, 1000 / 2, 0));
	pts.push_back(GePoint3d::create(0, -1000 / 2, 0));
	pLine = IGeCurveBase::createLineString(pts);
	outLines->push_back(pLine);
	assert(outLines->isClosedBoundaryType());
	parityRegion->add(outLines);

	GeVec3d veccc = GeVec3d::create(0, 0, 2000);
	GeExtrusionInfo extrData(parityRegion, veccc, true);
	IGeSolidBasePtr ptrExtrusion = IGeSolidBase::createGeExtrusion(extrData);
	
	ptrPhysicalGeometry->addGeSolidBase(*ptrExtrusion);
	return ptrPhysicalGeometry;
}