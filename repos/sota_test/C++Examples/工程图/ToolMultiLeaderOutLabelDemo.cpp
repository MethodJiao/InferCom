#include "stdafx.h"
#include "ToolMultiLeaderOutLabelDemo.h"
#include "PBBimDrawingCommon/PBimsPSLibAPIUtil.h"
#include "PBBimDrawingCommon/BPMultiLeaderOutLabel.h"

using namespace ::PBBim::PBBimDrawingCom;

void ToolMultiLeaderOutLabelDemo::createMultiLeaderOutLabel()
{
	BPViewportP vp = BPViewManager::getInstance().getActivedViewport();
	BPModelPtr rootModel = vp->getRootModel();
	PModelId rootModelId = rootModel->getModelId();
	double markSize = 3.5;
	BPProjectPtr _ptrProject = BPProject::getActiveProject();
	// 第一个
	BPMultiLeaderOutLabelPtr pLeader = BPMultiLeaderOutLabel::Create();
	pLeader->setTextString(L"第一行测试范例文字\r\n第二行测试范例文字\r\n显示基线%%130");
	pLeader->setAlignType(1);
	// 文字基线方向 -- 左
	pLeader->setTextDir(1);
	// 第一象限
	pLeader->setDrawingScale(1);
	pLeader->setLineStart(GePoint3d::create(0, 0, 0));
	pLeader->setLineEnd(GePoint3d::create(-20, 20, 0));
	// pLeader->setDrawTextLineLength(50);
	pLeader->setTextEndDis(60);
	pLeader->setTextLineLength(50);
	pLeader->setTextStartDis(10);
	pLeader->setMarkSize(markSize);
	// pLeader->setTextSweep(45);
	pLeader->addToProject(*BPProject::getActiveProject(), rootModelId);
	p3d::pvector<p3d::GePoint3d> pts = pLeader->getLabelPoints(*_ptrProject, *rootModel);
	pLeader->setVecPtMark(pts);
	pLeader->replaceInProject(*_ptrProject);
	// 文字基线方向 -- 右
	pts.clear();
	pLeader->setVecPtMark(pts);
	pLeader->setTextDir(0);
	pLeader->setLineStart(GePoint3d::create(0, 0, 0));
	pLeader->setLineEnd(GePoint3d::create(20, 20, 0));
	pLeader->addToProject(*BPProject::getActiveProject(), rootModelId);
	pts = pLeader->getLabelPoints(*_ptrProject, *rootModel);
	pLeader->setVecPtMark(pts);
	pLeader->replaceInProject(*_ptrProject);

	// 文字基线方向 -- 右
	pts.clear();
	pLeader->setVecPtMark(pts);
	pLeader->setTextDir(0);
	pLeader->setLineStart(GePoint3d::create(0, 0, 0));
	pLeader->setLineEnd(GePoint3d::create(20, -20, 0));
	// pLeader->setTextSweep(45);
	pLeader->addToProject(*BPProject::getActiveProject(), rootModelId);
	pts = pLeader->getLabelPoints(*_ptrProject, *rootModel);
	pLeader->setVecPtMark(pts);
	pLeader->replaceInProject(*_ptrProject);
	// 文字基线方向 -- 左
	pts.clear();
	pLeader->setVecPtMark(pts);
	pLeader->setTextDir(1);
	pLeader->setLineStart(GePoint3d::create(0, 0, 0));
	pLeader->setLineEnd(GePoint3d::create(-20, -20, 0));
	// pLeader->setTextSwedrawmodelDemoep(45);
	pLeader->addToProject(*BPProject::getActiveProject(), rootModelId);
	pts = pLeader->getLabelPoints(*_ptrProject, *rootModel);
	pLeader->setVecPtMark(pts);
	pLeader->replaceInProject(*_ptrProject);
}

AutoDoRegisterFunctionsBegin
//基线文字
BIMBase::BPToolsManager::registerFun("MultiLeaderOutLabelDemo", &ToolMultiLeaderOutLabelDemo::createMultiLeaderOutLabel);
AutoDoRegisterFunctionsEnd