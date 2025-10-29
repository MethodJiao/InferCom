#include "stdafx.h"
#include "ToolGeometryDemo.h"


ToolCreateGeometry::ToolCreateGeometry()
{
}


ToolCreateGeometry::~ToolCreateGeometry()
{
}

void ToolCreateGeometry::CreateGeometry()
{
	BPModelP pModel = BPViewManager::getInstance().getActivedViewport()->getTargetModel();
	//点
	GePoint3d pointS = GePoint3d::create(1, 1, 0);
	GePoint3d pointE = GePoint3d::create(8000, 8000, 0);
	GePoint3d pointB = GePoint3d::create(8000, 1, 0);
	GePoint3d pointT = GePoint3d::create(16000, 8000, 0);
	//线
	IGeCurveBasePtr ptrCurve = IGeCurveBase::createSegment(GeSegment3d::create(pointS, pointE));
	IGeCurveBasePtr ptrCurve1 = IGeCurveBase::createSegment(GeSegment3d::create(pointE, pointT));
	IGeCurveBasePtr ptrCurve2 = IGeCurveBase::createSegment(GeSegment3d::create(pointT, pointB));
	IGeCurveBasePtr ptrCurve3 = IGeCurveBase::createSegment(GeSegment3d::create(pointB, pointS));
	//面
	GeCurveArrayPtr ptrCurveList = GeCurveArray::create(GeCurveArray::BoundaryType::BOUNDARY_TYPE_Outer);
	if (ptrCurveList == nullptr)
		return;
	ptrCurveList->add(ptrCurve);
	ptrCurveList->add(ptrCurve1);
	ptrCurveList->add(ptrCurve2);
	ptrCurveList->add(ptrCurve3);
	//体
	GeVec3d v3d = GeVec3d::create(0, 0, 2000);
	GeExtrusionInfo extrusionInfo(ptrCurveList, v3d, true);
	IGeSolidBasePtr ptrSolid = IGeSolidBase::createGeExtrusion(extrusionInfo);
	if (ptrSolid == nullptr)
		return;
	BPGraphicsPtr ptrGrapic = pModel->createPhysicalGraphics();
	if (ptrGrapic.isNull())
		return;
	ptrGrapic->addGeSolidBase(*ptrSolid);
	ptrGrapic->save();

//拿到几何体的点，线，面
pvector <GeSolidLocationInfo::GeFaceIndices> indices;
//得到soild里面indices
ptrSolid->getFaceIndices(indices);
pvector<GeCurveArrayPtr> curearray;
//通过indices拿到对应的面
for (auto indice : indices)
{
	IGeometryPtr ptrGeom = ptrSolid->getFace(indice);
	if(ptrGeom.isNull())
		continue;

	GeCurveArrayPtr ptrCv = ptrGeom->getAsGeCurveArray();
	if(ptrCv.isNull())
		continue;
	curearray.push_back(ptrCv);

}
for (int i = 0; i < curearray.size(); i++)
{
	//拿出每一个面
	GeCurveArrayPtr ptrCu = curearray[i];
	if (ptrCu != nullptr)
	{
		int size = ptrCu->size();
		for (int j = 0; j < size; j++)
		{
			IGeCurveBase::CurveBaseType type = ptrCu->at(j)->getCurveBaseType();
			//拿出每一个面中线的信息
			if (type == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_Segment)
			{
				IGeCurveBasePtr ptrCurbase = ptrCu->at(j);
				if(ptrCurbase.isNull())
					continue;
				GePoint3d pointA = GePoint3d::create(0,0,0);
				GePoint3d pointB = GePoint3d::create(0, 0, 0);
				ptrCurbase->getStartEndPoint(pointA, pointB);						
			}
			else if (type == IGeCurveBase::CurveBaseType::CURVE_BASE_TYPE_LineString)
			{
				IGeCurveBasePtr ptrCurbase = ptrCu->at(j);
				if(ptrCurbase.isNull())
					continue;				
				const pvector<GePoint3d> *pts = ptrCurbase->getLineStringCP();				
			}
		}
	}
}

}


/**
@brief   创建样条曲线
@file   
@author  北京构力科技有限公司
@date    2021.08.03
*/

void funBYT()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return;

	pvector<GePoint3d> points;
	pvector<GePoint3d> controlPoints = {
		GePoint3d::createByZero(),
		GePoint3d::create(1000, 2000, 3000),
		GePoint3d::create(1500, 3000, 4000),
		GePoint3d::create(3000, 4500, 2000),
		GePoint3d::create(6000, 6000, 6000)
	};

	if (!BIMBase::SolidCore::BPCurveApproximationFunction::createGeBsplinePoints(points, controlPoints, 4, 100))
		return;

	GeCurveArrayPtr ptrline = GeCurveArray::createLinestringArray(points);
	if (ptrline.isNull())
		return;

	BPGraphicsPtr ptrGrapic = pModel->createPhysicalGraphics();
	if (ptrGrapic.isNull())
		return;

	if (0 != ptrGrapic->addGeCurveArray(*ptrline))
		return;

	ptrGrapic->finish();
	ptrGrapic->save();
};

/**
@brief   实体过滤
@file
@author  北京构力科技有限公司
@date    2021.08.03
*/
void DemoElementFilter()
{
	BPEntityArray elements;
	GeRange3d range = GeRange3d::create(GePoint3d::createByZero(), GePoint3d::create(10000, 10000, 1000));
	BPProjectP pProject = BIMBase::Core::BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == NULL)
	{
		return;
	}

	if (0 != BPEntityUtil::getEntitiesByRange(elements, range, *pProject))
	{
		AfxMessageBox(L"特定范围未找到对象");
		return;
	}
		
	wstring wsOutput = L"找到";
	wsOutput += to_wstring(elements.getCount()) + L"对象";

	AfxMessageBox(wsOutput.c_str());
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(L"elefilterDemo", DemoElementFilter);
BPToolsManager::registerFun("bytDemo", &funBYT);
BPToolsManager::registerFun("geometryDemo", ToolCreateGeometry::CreateGeometry);
AutoDoRegisterFunctionsEnd