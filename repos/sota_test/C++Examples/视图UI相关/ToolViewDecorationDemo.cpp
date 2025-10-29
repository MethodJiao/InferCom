#include "stdafx.h"
#include "ToolViewDecorationDemo.h"



ViewDecorationDemo::ViewDecorationDemo()
{
}


ViewDecorationDemo::~ViewDecorationDemo()
{
}

bool ViewDecorationDemo::_drawDecoration(BPViewportR viewport)
{
	int nView = viewport.getViewNumber();

	BPViewDrawP pOutput = viewport.getIViewDraw();
	if (pOutput == nullptr)
		return false;

	//随着屏幕缩放可改变--------------------改变大小及位置
	GePoint3d ptOri = GePoint3d::create(0, 0, 0);
	GeVec3d vec = GeVec3d::create(1, 1, 0);
	GePoint3d ptEnd = ptOri + vec * 2000;
	IGeCurveBasePtr ptrLine = IGeCurveBase::createSegment(GeSegment3d::create(ptOri, ptEnd));
	if (ptrLine.isNull())
		return false;

	GeCurveArrayPtr ptrCurve = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer, ptrLine);
	if (ptrCurve.isNull())
		return false;

	pOutput->setSymbology(RGB(255, 0, 0), RGB(255, 0, 0), 2, 0);
	pOutput->drawGeCurveArray(*ptrCurve, true);


	//不随屏幕缩放而改变------------------------固定大小及位置
	p3d::GeRect rect;
	viewport.getViewRect(rect);
	
	//获取线条位置
	GePoint3d ptPlaceStart = viewport.viewToWorld(GePoint3d::create(rect.left() + (rect.right() - rect.left()) / 2, rect.bottom() - (rect.bottom() - rect.top()) / 20, 0));
	GePoint3d ptPlaceEnd = viewport.viewToWorld(GePoint3d::create(rect.left(), rect.bottom() - (rect.bottom() - rect.top()) / 20, 0));

	IGeCurveBasePtr Line3 = IGeCurveBase::createSegment(GeSegment3d::create(ptPlaceStart, ptPlaceEnd));
	if (Line3.isNull())
		return false;

	GeCurveArrayPtr ptrCurve3 = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer, Line3);
	if (ptrCurve3.isNull())
		return false;

	pOutput->setSymbology(RGB(0, 255, 0), RGB(0, 255, 0), 2, 0);
	pOutput->drawGeCurveArray(*ptrCurve3, true);


	//不随屏幕缩放而改变------------------------固定大小
	//获取旋转矩阵
	GeRotMatrix rotMatrix = GeRotMatrix::createIdentityMatrix();		
	viewport.getRotMatrix(rotMatrix);
	GeVec3d vtX, vtY;
	rotMatrix.getRow(vtX, 0);
	rotMatrix.getRow(vtY, 1);
	GePoint3d ptOri2 = GePoint3d::create(0, 500, 0);
	GePoint3d ptOri2View = viewport.worldToView(ptOri2);
	GePoint3d ptEnd2View = ptOri2View - vtY * 50 + vtX*50;

	GePoint3d ptEnd2 = viewport.viewToWorld(ptEnd2View);


	IGeCurveBasePtr ptrLine2 = IGeCurveBase::createSegment(GeSegment3d::create(ptOri2, ptEnd2));
	if (ptrLine2.isNull())
		return false;

	GeCurveArrayPtr ptrCurve2 = GeCurveArray::create(GeCurveArray::BOUNDARY_TYPE_Outer, ptrLine2);
	if (ptrCurve2.isNull())
		return false;

	pOutput->setSymbology(RGB(0, 0, 255), RGB(0, 0, 255), 2, 0);
	pOutput->drawGeCurveArray(*ptrCurve2, true);

    //如果是多视口，则返回false
	return true;
}

static ViewDecorationDemo g_ViewDecorationDemo;
void ViewDecorationDemo::begin()
{
	BPViewManager::getInstance().registerViewDecoration(&g_ViewDecorationDemo);
}

void ViewDecorationDemo::end()
{
	BPViewManager::getInstance().unregisterViewDecoration(&g_ViewDecorationDemo);
}

ViewDecorationDemo& ViewDecorationDemo::get()
{
	return g_ViewDecorationDemo;
}

//--------------------------------------------------------------------------------------------------

void ToolLayoutViewDecorationDemo::_onPostInstall()
{
	ViewDecorationDemo::get().begin();
}

void ToolLayoutViewDecorationDemo::_onRestartTool()
{

}

bool ToolLayoutViewDecorationDemo::_onDataButton(BPBaseButtonEventCP)
{
	return true;
}

bool ToolLayoutViewDecorationDemo::_onResetButton(BPBaseButtonEventCP)
{
	ViewDecorationDemo::get().end();
	_exitTool();
	return true;
}

void ToolLayoutViewDecorationDemo::_onDynamicFrame(BPBaseButtonEventCP)
{

}

bool ToolLayoutViewDecorationDemo::_onModelMotion(BPBaseButtonEventCP ev)
{
	return false;
}

ToolLayoutViewDecorationDemo::ToolLayoutViewDecorationDemo()
{

}

ToolLayoutViewDecorationDemo::~ToolLayoutViewDecorationDemo()
{

}


BPTool* CreateDecorationTool()
{
	ToolLayoutViewDecorationDemo* tool = new ToolLayoutViewDecorationDemo();
	return tool;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerTool("layoutDecorationDemo", &CreateDecorationTool);
AutoDoRegisterFunctionsEnd