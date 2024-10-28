using Microsoft.AspNetCore.Mvc;
using VertoTest.Models;

namespace VertoTest.Services
{
    public interface ContentPanelInterface
    {
        Task<bool> Delete(ContentPanels content);
        Task<bool> Edit(ContentPanels content);
        Task<List<ContentPanels>> List();
        Task Add(ContentPanels content);
        Task<ContentPanels?> GetViaId(int id);
        Task<bool> Exists(int sliderItemid);
    }
}
