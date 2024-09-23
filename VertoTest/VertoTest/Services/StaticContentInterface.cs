using VertoTest.Context;
using VertoTest.Models;

namespace VertoTest.Services
{
    public interface StaticContentInterface
    {
        Task<int> Add(StaticContent content);
        Task<List<StaticContent>> List();
        Task<List<StaticContent>> ListWithClass(String className);
        Task<bool> Edit(StaticContent content);
        Task<bool> Exists(int id);

        Task<bool> Delete(StaticContent content);
        Task<StaticContent?> GetViaId(int id);


    }
}
