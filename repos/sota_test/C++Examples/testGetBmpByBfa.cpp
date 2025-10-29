#include "stdafx.h"

BPGraphicsPtr getGraphicFromDataList(BPDataList insList, BPProjectP project)
{
	PPCGraphics taiji;
	for (auto& ins : insList)
	{
		BPDataKey key = ins->getDataKey();
		std::set<BPEntityId> entitySet;
		BPDataUtil::getAllBindingEntityFromData(entitySet, key, project);
		for (auto entiId : entitySet)
		{
			BPEntity entity(entiId, *project);
			if (!entity.isValid())
				continue;
			BPGraphicsPtr ptrGra = BPGraphics::ReadPhysicalGraphics(entity);
			if (ptrGra.isNull())
				continue;
			if (ptrGra->getSymbology().color > 255)
			{
				BPColorDef colorrgb;
				if (P3DStatus::SUCCESS == BPColorUtil::extractEntityColor(&colorrgb, nullptr, nullptr, ptrGra->getSymbology().color, *project))
				{
					ptrGra->getSymbology().color = BPColorUtil::getEntityColor(colorrgb, *BPApplication::getInstance().getProjectManager()->getActiveProject(), true);
				}
			}
			for (auto& it : *ptrGra)
			{
				if (it->getSymbology().color > 255)
				{
					BPColorDef colorrgb;
					if (P3DStatus::SUCCESS == BPColorUtil::extractEntityColor(&colorrgb, nullptr, nullptr, it->getSymbology().color, *project))
					{
						it->getSymbologyR().color = BPColorUtil::getEntityColor(colorrgb, *BPApplication::getInstance().getProjectManager()->getActiveProject(), true);
					}
				}
			}
			taiji.add(ptrGra);
		}
	}
	return taiji.get();
};
//根据bfa模型生成三张缩略图，根据尺寸不同分为小图、大图、正常图
static void getBmpByBfa()
{
	//路径为需要生成缩略图的bfa模型所在路径
	std::wstring path = L"PLATFORM\\BPGroup\\ComponentLib\\变电\\电气\\变压器\\三绕组变压器\\110kV变压器模型15.bfa";
	std::wstring bfaPath = BIMBase::Core::BPApplication::getInstance().getAppPath().c_str() + path;
	CString strPath = bfaPath.c_str();
	BPProjectP pProject = ::BIMBase::Core::BPProject::getActiveProject();
	if (pProject == nullptr)
		return ;
	int nIndex = strPath.ReverseFind('.');
	CString strBmp = strPath.Left(nIndex);
	strBmp += L".bmp";

	//判断图是否已经存在,如果已经存在缩略图，则直接返回
	WIN32_FILE_ATTRIBUTE_DATA attrs = { 0 };
	if (0 != GetFileAttributesEx(strBmp, GetFileExInfoStandard, &attrs))
		return ;

	// 如果不存在缩略图，重新生成
	strBmp = strPath.Left(nIndex);
	CString strBmp_s = strBmp + L"_s.bmp";//小图
	CString strBmp_b = strBmp + L"_b.bmp";//大图
	strBmp += L".bmp";//正常图

	P3DFileName bfaFilePath = P3DFileName(strPath.GetString());
	P3DStatus t_status;
	BPProjectPtr bfaProject = BPApplication::getInstance().getProjectManager()->openProject(t_status, bfaFilePath, BPProject::BPOpenMode::enReadWrite);
	if (t_status != P3DStatus::SUCCESS)
		return ;
	if (bfaProject.isNull())
		return ;

	BPProjectHandle bfaHandle = bfaProject->getProjectHandle();
	BPDataList insList;
	BPDataUtil::getAllDataFromProject(insList, *bfaProject);

	BPGraphicsPtr bfa_gra = getGraphicFromDataList(insList, bfaProject.get());
	if (bfa_gra->isEmpty())
	{
		PBBim::PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, Utf8String(L"getGraphicFromDataList返回空，无法生成预览图！").c_str());
		return ;
	}
	BPEntityR entity = bfa_gra->getEntityR();
	if (!entity.isValid())
	{
		PBBim::PBMessageCenter::Send(PBBim_MESSAGE_ToolPrompt, Utf8String(L"预览图的BPEntityR为空，生成图片失败！").c_str());
		return ;
	}
	pvector<BPEntityP>entitys;
	entitys.push_back(&entity);
	int t_width_s = 30;//小图尺寸
	int t_height_s = 29;
	int t_width = 500;//正常图尺寸
	int t_height = 500;
	int t_width_b = 1000;//大图尺寸
	int t_height_b = 1000;

	GeRotMatrix rot;
	rot.setByStandardViewMartix(7);
	if (!Core::BPViewport::entityToBitmap(entitys, BPStandardView::enIsometric, BPRenderMode::enSmoothShade, t_width, t_height, strBmp.GetString(), rot)
		&& !Core::BPViewport::entityToBitmap(entitys, BPStandardView::enIsometric, BPRenderMode::enSmoothShade, t_width_s, t_height_s, strBmp_s.GetString(), rot)
		&& !Core::BPViewport::entityToBitmap(entitys, BPStandardView::enIsometric, BPRenderMode::enSmoothShade, t_width_b, t_height_b, strBmp_b.GetString(), rot))
		return ;
	return ;
}

AutoDoRegisterFunctionsBegin
BIMBase::BPToolsManager::registerFun("testgetbmp", &getBmpByBfa);
AutoDoRegisterFunctionsEnd

