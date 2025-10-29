#include "stdafx.h"

void funCurveArrayOffset()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return;

	for (int i = 0; i < 30; i++)
	{
		for (int j = 0; j < 30; j++)
		{
			BPGraphicsPtr ptrGraphics = pModel->createPhysicalGraphics();
			if (ptrGraphics.isNull())
				return;

			pvector<GePoint3d> pts = {
				GePoint3d::createByZero() + GePoint3d::create(500 * i, 500 * j, 0),
				GePoint3d::create(200, 0, 0) + GePoint3d::create(500 * i, 500 * j, 0),
				GePoint3d::create(0, 200, 0) + GePoint3d::create(500 * i, 500 * j, 0)
			};
			GeCurveArrayPtr ptrCA = GeCurveArray::createLinestringArray(pts, GeCurveArray::BOUNDARY_TYPE_Open);

			GeCurveOffsetOptions offsetOptionOut(20);
			double dCA = offsetOptionOut.getChamferAngle();
			offsetOptionOut.setChamferAngle(10);

			double dTo = offsetOptionOut.getTolerance();
			offsetOptionOut.setTolerance(j);

			double dAng = offsetOptionOut.getArcAngle();


			GeCurveArrayPtr ptrCAOffsetOut = ptrCA->cloneOffsetCurves2D(offsetOptionOut);
			if (ptrCAOffsetOut.isNull())
				return;

			GeCurveOffsetOptions offsetOptionIn(-20);
			offsetOptionIn.setChamferAngle(10);
			offsetOptionIn.setTolerance(j);
			GeCurveArrayPtr ptrCAOffsetIn = ptrCA->cloneOffsetCurves2D(offsetOptionIn);
			if (ptrCAOffsetIn.isNull())
				return;

			ptrGraphics->addGeCurveArray(*ptrCA);
			ptrGraphics->addGeCurveArray(*ptrCAOffsetOut);
			ptrGraphics->addGeCurveArray(*ptrCAOffsetIn);

			wstring ws = std::to_wstring(i) + L", " + std::to_wstring(j);
			/*ptrGraphics->addTextString(ptrDemo);*/
			ptrGraphics->finish();
			ptrGraphics->save();
		}

	}
}

void funCurveArrayOpenOffset()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return;

	for (int i = 0; i < 30; i++)
	{
		for (int j = 0; j < 30; j++)
		{
			BPGraphicsPtr ptrGraphics = pModel->createPhysicalGraphics();
			if (ptrGraphics.isNull())
				return;

			pvector<GePoint3d> pts = {
				GePoint3d::createByZero() + GePoint3d::create(1000 * i, 1000 * j, 0),
				GePoint3d::create(200, 0, 0) + GePoint3d::create(1000 * i, 1000 * j, 0),
				GePoint3d::create(500, 200, 0) + GePoint3d::create(1000 * i, 1000 * j, 0),
				GePoint3d::create(300, 300, 0) + GePoint3d::create(1000 * i, 1000 * j, 0),
				GePoint3d::create(600, 500, 0) + GePoint3d::create(1000 * i, 1000 * j, 0),
				GePoint3d::create(0, 700, 0) + GePoint3d::create(1000 * i, 1000 * j, 0),
			};
			GeCurveArrayPtr ptrCA = GeCurveArray::createLinestringArray(pts, GeCurveArray::BOUNDARY_TYPE_Open);

			GeCurveOffsetOptions offsetOptionOut(20);
			double dCA = offsetOptionOut.getChamferAngle();
			offsetOptionOut.setChamferAngle(10);

			double dTo = offsetOptionOut.getTolerance();
			offsetOptionOut.setTolerance(j);

			double dAng = offsetOptionOut.getArcAngle();


			GeCurveArrayPtr ptrCAOffsetOut = ptrCA->cloneOffsetCurves2D(offsetOptionOut);
			if (ptrCAOffsetOut.isNull())
				return;

			GeCurveOffsetOptions offsetOptionIn(-20);
			offsetOptionIn.setChamferAngle(80);
			offsetOptionIn.setTolerance(j);
			GeCurveArrayPtr ptrCAOffsetIn = ptrCA->cloneOffsetCurves2D(offsetOptionIn);
			if (ptrCAOffsetIn.isNull())
				return;

			ptrGraphics->addGeCurveArray(*ptrCA);
			ptrGraphics->addGeCurveArray(*ptrCAOffsetOut);
			ptrGraphics->addGeCurveArray(*ptrCAOffsetIn);

			wstring ws = std::to_wstring(i) + L", " + std::to_wstring(j);
			ptrGraphics->finish();
			ptrGraphics->save();
		}       

	}
}

void funCurveArrayChamferOffset()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return;

	BPGraphicsPtr ptrGraphics = pModel->createPhysicalGraphics();
	if (ptrGraphics.isNull())
		return;

	pvector<GePoint3d> pts = {
		GePoint3d::createByZero(),
		GePoint3d::create(1000, 0, 0),
		GePoint3d::create(1000, 1000, 0),
	};
	GeCurveArrayPtr ptrCA = GeCurveArray::createLinestringArray(pts, GeCurveArray::BOUNDARY_TYPE_Open);

	GeCurveArrayPtr ptrCAOffsetOut = ptrCA->cloneWithEllipseChamfering(200);
	if (ptrCAOffsetOut.isNull())
		return;

	BPGraphicsPtr ptrGraphics2 = pModel->createPhysicalGraphics();
	if (ptrGraphics2.isNull())
		return;
	int nn = ptrCAOffsetOut->size();
	GeCurveArrayPtr ptrnn = GeCurveArray::create( GeCurveArray::BOUNDARY_TYPE_None);
	for (int i = 0; i < nn; i++)
	{
		IGeCurveBasePtr type = ptrCAOffsetOut->at(i);
		IGeCurveBasePtr curve = IGeCurveBase::createProportCurve(type.get(), 0, 0.5);
		double dd = 0;
		curve->getLength(dd);
		GePoint3d ps, pe;
		curve->getStartEndPoint(ps, pe);
		ptrnn->add(curve);
	}
	ptrGraphics2->addGeCurveArray(*ptrnn);
	

	GeCurveOffsetOptions offsetOptionIn(-20);
	offsetOptionIn.setTolerance(200);
	GeCurveArrayPtr ptrCAOffsetIn = ptrCA->cloneOffsetCurves2D(offsetOptionIn);
	if (ptrCAOffsetIn.isNull())
		return;

	ptrGraphics->addGeCurveArray(*ptrCAOffsetOut);
	ptrGraphics->addGeCurveArray(*ptrCAOffsetIn);
	ptrGraphics2->addGeCurveArray(*ptrCAOffsetIn);

	ptrGraphics->finish();
	ptrGraphics->save();
	ptrGraphics2->finish();
	BPEntityId idd =  ptrGraphics2->save();
}

void funCurveProportDemo()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return;
	BPGraphicsPtr ptrGraphics2 = pModel->createPhysicalGraphics();
	if (ptrGraphics2.isNull())
		return;
	
	IGeCurveBasePtr curvebase = IGeCurveBase::createSegment(GeSegment3d::create(GePoint3d::createByZero(), GePoint3d::create(1000, 1000, 0)));
	IGeCurveBasePtr curve = curvebase->cloneByProportParas(0, 0.5, false);
	double dd = 0;
	curve->getLength(dd);
	GePoint3d ps, pe;
	curve->getStartEndPoint(ps, pe);
	ptrGraphics2->addGeCurve(*curve);
	ptrGraphics2->save();
}


void funCurveArrayAreaOffset()
{
	BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
	if (pProject == nullptr)
		return;

	BPModelBaseP pModel = pProject->getActiveModel();
	if (pModel == nullptr)
		return;
	for (int i = 0; i < 30; i++)
	{
		for (int j = 0; j < 30; j++)
		{
			BPGraphicsPtr ptrGraphics = pModel->createPhysicalGraphics();
			if (ptrGraphics.isNull())
				return;

			pvector<GePoint3d> pts = {
				GePoint3d::createByZero() + GePoint3d::create(500 * i, 500 * j, 0),
				GePoint3d::create(200, 0, 0) + GePoint3d::create(500 * i, 500 * j, 0),
				GePoint3d::create(0, 200, 0) + GePoint3d::create(500 * i, 500 * j, 0)
			};
			GeCurveArrayPtr ptrCA = GeCurveArray::createLinestringArray(pts, GeCurveArray::BOUNDARY_TYPE_Outer);

			GeCurveOffsetOptions offsetOptionOut(20);
			offsetOptionOut.setChamferAngle(0.1 * i);
			offsetOptionOut.setTolerance(j);
			double dto = offsetOptionOut.getTolerance();
			double dang = offsetOptionOut.getArcAngle();
			double dca = offsetOptionOut.getChamferAngle();

			GeCurveArrayPtr ptrCAOffsetOut = ptrCA->cloneAreaOffset(offsetOptionOut);
			if (ptrCAOffsetOut.isNull())
				return;

			GeCurveOffsetOptions offsetOptionIn(-20);
			offsetOptionIn.setChamferAngle(0.1 * i);
			offsetOptionIn.setTolerance(j);
			GeCurveArrayPtr ptrCAOffsetIn = ptrCA->cloneAreaOffset(offsetOptionIn);
			if (ptrCAOffsetIn.isNull())
				return;

			ptrGraphics->addGeCurveArray(*ptrCA);
			ptrGraphics->addGeCurveArray(*ptrCAOffsetOut);
			ptrGraphics->addGeCurveArray(*ptrCAOffsetIn);

			ptrGraphics->finish();
			ptrGraphics->save();
		}
	}
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun("funCAODemo", &funCurveArrayOffset);
BPToolsManager::registerFun("funCAOODemo", &funCurveArrayOpenOffset);
BPToolsManager::registerFun("funCAAODemo", &funCurveArrayAreaOffset);
BPToolsManager::registerFun("funChamferDemo", &funCurveArrayChamferOffset);
BPToolsManager::registerFun("funCurveProportDemo", &funCurveProportDemo);
AutoDoRegisterFunctionsEnd
