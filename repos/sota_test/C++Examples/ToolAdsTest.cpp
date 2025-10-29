#include "stdafx.h"
#include "ToolAdsTest.h"
#include "P3DCore\adsarxfunc\ADSARXFUN.h"

void ToolAdsTest::AdsLine()
{
    BPProjectP pProject = BPProject::getActiveProject();
    if (pProject == nullptr)
        return;
    BPModelBaseP pModel = pProject->getActiveModel();
    if (pModel == nullptr)
        return;
    BPGraphicsPtr ptrGraphic = pModel->createPhysicalGraphics();
    if (ptrGraphic.isNull())
        return;

    double dFirst[3], dSecond[3];
    int status = getPoint(nullptr, "指定第一个点:", dFirst);
    GePoint3d ptFirst = GePoint3d::create(dFirst[0], dFirst[1], dFirst[2]);
    const double* ptNow[3] = {&ptFirst.x, &ptFirst.y, &ptFirst.z};
    BPViewportP pViewPort = BPViewManager::getInstance().getActivedViewport();
    IDragCallback dcb;
    dcb.OnDynamicDraw(*ptNow, (IDrawViewport&)pViewPort);

    char opr[32] = ("O F E C");
    //initget(128, _T("O F E C"));
    int rek = getKeyword(("\n请选择脚本的类型:[管件(F)/设备(E)/物料(C)] "), opr);

    initGet(InputOptions::rsg_other, "D");
    //int res = getString(true, "指定第二个点或[尺寸(D)]:", "");  //输入数字可以，仅可输入一个字母，输入其他字母啥的就跳前面去了
    int nRes = getPoint("指定第二个点或[尺寸(D)]:", dcb, dSecond);
    GePoint3d ptSecond = GePoint3d::create(pSecond[0], pSecond[1], pSecond[2]);
    ptrGraphic->AddCurvePrimitive(*IGeCurveBase::createSegment(GeSegment3d::create(ptFirst, ptSecond)));
    ptrGraphic->save();

    //设置ac轴位置
    BPTempUcsMngr::activate();
    BPTempUcsMngr::setOrigin(GePoint3d::create(1000,1000,1000));

    //switch (res)
    //{
    //case rtOk:
    //{
    //    GePoint3d ptSecond = GePoint3d::create
    //    (pSecond[0], pSecond[1], pSecond[2]);
    //    ptrGraphic->AddCurvePrimitive(*IGeCurveBase::createSegment
    //    (GeSegment3d::create(ptFirst, ptSecond)));
    //    ptrGraphic->save();
    //}
    //break;
    //case rtCancel:
    //    break;
    //case rtError:
    //    break;
    //case rtKeyword:
    //{
    //    char cInput[100];
    //    getInput(cInput);
    //    if (strcmp(cInput, "D") != 0 && strcmp(cInput, "d") != 0)
    //        return;
    //    initGet(InputOptions::rsg_other, "");
    //    int nLength = 0, nWidth = 0;
    //    int nRes = getInt("指定X方向增量<70>:", nLength);
    //    //initGet(InputOptions::rsg_other, "");
    //    nRes = getInt("指定Y方向增量<70>:", nWidth);//这一步会有问题，追踪器面板出不来，Esc后到下一行
    //    GePoint3d ptSecond = GePoint3d::create
    //    (pFirst[0] + nLength, pFirst[1] + nWidth, pFirst[2]);
    //    ptrGraphic->AddCurvePrimitive(*IGeCurveBase::createSegment
    //    (GeSegment3d::create(ptFirst, ptSecond)));
    //    ptrGraphic->save();
    //}
    //break;
    //case rtNone:
    //    break;
    //default:
    //    break;
    //}
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(_T("ToolAdsTest"), &ToolAdsTest::AdsLine);
AutoDoRegisterFunctionsEnd
