#include "stdafx.h"
#include "ToolLayoutCubeDemo.h"

#include "LibXL\libxl.h"
#pragma comment(lib,"libxl.lib")

using namespace libxl;

char* w2c(char* pcstr, const wchar_t* pwstr, size_t len)
{
    int nlength = wcslen(pwstr);
    //获取转换后的长度
    int nbytes = WideCharToMultiByte(0, 0, pwstr, nlength, NULL, 0, NULL, NULL);
    if (nbytes > len)   nbytes = len;
    // 通过以上得到的结果，转换unicode 字符为ascii 字符
    WideCharToMultiByte(0, 0, pwstr, nlength, pcstr, nbytes, NULL, NULL);
    return pcstr;
}

void importexcelDemo()
{
    BPProjectP pProject = BPApplication::getInstance().getProjectManager()->getMainProject();
    if (pProject == nullptr)
        return;
    PModelId modelId = pProject->getDefaultModelId();
    Book* book = xlCreateXMLBook();
    book->setKey(L"Halil Kural", L"windows-2723210a07c4e90162b26966a8jcdboe");
    //创建新excel进行写数据
    Book* book1 = xlCreateXMLBook();
    book1->setKey(L"Halil Kural11", L"windows-2723210a07c4e90162b26966a8jcdboe");
    Sheet* shet = book1->addSheet(L"使用libxl库新建的表");
    Format* formdata = book1->addFormat();
    //把长宽高各自包含的数据保存
    vector<double> vecLength;
    vector<double> vecWidth;
    vector<double> vecHeigth;
	p3d::PString appPathroot = BPApplication::getInstance().getAppPath();
	p3d::PString outfileName = appPathroot + L"..\\..\\C++Examples-Plugin\\Examples\\Demofile";
	p3d::PString filefullName = appPathroot + L"..\\..\\C++Examples-Plugin\\Examples\\Demofile\\xlsx读取测试文件.xlsx";
	if (book->load(filefullName.c_str()))
    {
        Sheet* sheetread = book->getSheet(0);
        if (sheetread)
        {
            for (int i = 1; i < 4; i++)//列
            {
                for (int j = 0; j < 3; j++)//行
                {
                    CellType celltype = sheetread->cellType(j, i);
                    Format* format = sheetread->cellFormat(j, i);
                    if (celltype == CELLTYPE_STRING)
                    {
                        const wchar_t* t = sheetread->readStr(j, i);
                        wstring strName(t);
                        //只写表头的数据字段，数据后面会改变
                        if(j==0)//只看第0行的数据
                            shet->writeStr(j+1, i, t, formdata);
                      
                    }
                    else if (celltype == CELLTYPE_NUMBER)
                    {
                        double result = sheetread->readNum(j, i);
                        CellType celltype = sheetread->cellType(0, i);
                        if (celltype == CELLTYPE_STRING)
                        {
                            const wchar_t* t = sheetread->readStr(0, i);
                            wstring strName(t);
                            if (strName == L"Length")
                            {
                                vecLength.push_back(result);
                            }
                            else if (strName == L"Width")
                            {
                                vecWidth.push_back(result);
                            }
                            else if (strName == L"Height")
                            {
                                vecHeigth.push_back(result);
                            }


                        }
                        
                    }
                    else if (celltype == CELLTYPE_BLANK)
                    {
                        
                    }
                    else if (celltype == CELLTYPE_EMPTY)
                    {
                       
                    }
                }
                cout << endl;
            }
        }
        DemoObject::CubeDemoP cube = new DemoObject::CubeDemo();
        for (int k = 0; k < vecLength.size();k++)
        {
            
            cube->setLength(vecLength.at(k));
            cube->setWidth(vecWidth.at(k));
            cube->setHeight(vecHeigth.at(k));
            //增加构件到工程中
            if (SUCCESS != cube->addToProject(*pProject, modelId))
            {
                AfxMessageBox(L"Can not add to project!");
            }

        }
        //获取数据库里的值，然后修改长度
        std::vector<BPDataKey> dataKeys;
       
        BPDataUtil::getDataFromSchemaNameWhere_Quick(dataKeys,PBM_SCHEMA_Demo, PBM_CLASS_CUBE_Demo,L"Length", BPValue(cube->getLength()),*pProject);
       //拿到datakey，修改对象长度，保存到数据库
        for (int i = 0; i < dataKeys.size();i++)
        {
            BPDataPtr ptrData = BPDataUtil::getDataByKey(dataKeys[i], *pProject);
            if (ptrData == nullptr)
                continue;
            DemoObject::CubeDemoP cube = DemoObject::CubeDemo::create(*ptrData);
            if (cube == nullptr)
                continue;
            cube->setLength(5000);
            if (SUCCESS != cube->replaceInProject(*pProject))
            {
                AfxMessageBox(L"replace fail!");
            }
        }
        book->release();
    }
    //将修改完后的数据输出到新的excel中
    
    BPDataList instanceList;
    P3DStatus status = BPDataUtil::getDataFromSchemaName(instanceList, PBM_SCHEMA_Demo, PBM_CLASS_CUBE_Demo, *pProject);
    if (status != 0 || instanceList.empty())
        return;
    vecLength.clear();
    vecWidth.clear();
    vecHeigth.clear();

    for (const auto& instance : instanceList)
    {
        if (!instance.isValid())
            continue;
        DemoObject::CubeDemoP cube = DemoObject::CubeDemo::create(*instance);
        if (cube == nullptr)
            continue;
        vecLength.push_back(cube->getLength());
        vecWidth.push_back(cube->getWidth());
        vecHeigth.push_back(cube->getHeight());
    }
    //拿到新建excel里写进去的属性字段然后把从p3d里拿到的数值填进去
    Sheet* sheetread1 = book1->getSheet(0);
    if (sheetread1)
    {
        for (int i = 1; i < 4; i++)//列
        {
            
                CellType celltype = sheetread1->cellType(1, i);
                Format* format = sheetread1->cellFormat(1, i);
                if (celltype == CELLTYPE_STRING)
                {
                    const wchar_t* t = sheetread1->readStr(1, i);
                    wstring strName(t);
                    for (int j = 2; j <  vecLength.size() + 2; j++)//行
                    {
                        if (strName == L"Length")
                        {
                            double len = vecLength.at(j - 2);

                            shet->writeNum(j, i, len, formdata);
                        }
                        else if (strName == L"Width")
                        {
                            double width = vecWidth.at(j - 2);
                            shet->writeNum(j, i, width, formdata);
                        }
                        else if (strName == L"Height")
                        {
                            double hei = vecHeigth.at(j - 2);
                            shet->writeNum(j, i, hei, formdata);
                        }

                    }
                    
                }
        }
    }
	p3d::PString outfilepath = outfileName + L"\\xlsx新建测试文件.xlsx";
	book1->save(outfilepath.c_str());
	AfxMessageBox(L"生成表格路径在C++Examples-Plugin\\Examples\\Demofile下");
    book1->release();
    return ;
}

AutoDoRegisterFunctionsBegin
BPToolsManager::registerFun(_T("importexcelDemo"), &importexcelDemo);
AutoDoRegisterFunctionsEnd